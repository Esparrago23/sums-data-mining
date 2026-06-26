"""
load_to_bd.py — Carga de familias sintéticas a la BD real (vía API)
===================================================================
Inserta cada familia de `families_full.json` en la base de datos real del SUMS
usando el endpoint que YA hace toda la descomposición a las 55 tablas:

        POST {base_url}/cedulas/captura-completa

Por qué vía API y no SQL directo:
  - Reutiliza la lógica YA probada del backend (CapturaCompletaCedulaUseCase):
    crea núcleo, dirección, vivienda, integrantes + tablas satélite, vacunación,
    y resuelve los catálogos automáticamente (findOrCreateCatalog).
  - Funciona contra la API desplegada o contra una API local (solo cambia --base-url).
  - No requiere credenciales de BD ni replicar el mapeo en Python.

El endpoint NO requiere autenticación (no hay middleware global; solo /users está
protegido). Aun así, se permite --user/--password o --token por si eso cambia.

REANUDABLE: lleva un log JSONL (data/processed/load_results.jsonl). Si se corta o se
vuelve a correr, omite las familias ya cargadas (por índice) → no duplica.

USO:
  # Prueba con 5 familias contra la API local:
  python load_to_bd.py --base-url http://localhost:3000/sums --limit 5

  # Carga completa contra la API desplegada:
  python load_to_bd.py --base-url https://api-sums.troy.engineer/sums

Requiere: requests  (pip install requests)
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Falta 'requests'. Instala con: pip install requests")


def cargar_familias(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def cargar_indices_hechos(log_path):
    """Lee el log JSONL y devuelve el set de índices ya cargados con éxito."""
    done = set()
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get('ok'):
                        done.add(rec['idx'])
                except json.JSONDecodeError:
                    continue
    return done


def ping(session, base_url, timeout):
    try:
        r = session.get(f"{base_url}/ping", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def login(session, base_url, usuario, contrasena, timeout):
    r = session.post(
        f"{base_url}/login",
        json={'nombre_usuario': usuario, 'contrasena': contrasena},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    token = data.get('token')
    if not token:
        raise RuntimeError(f"Login sin token en la respuesta: {data}")
    return token


def post_familia(session, base_url, payload, timeout, max_retries=3):
    """POST con reintentos en 5xx / timeouts. Devuelve (ok, data_o_error)."""
    url = f"{base_url}/cedulas/captura-completa"
    last_err = None
    for intento in range(1, max_retries + 1):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            if r.status_code in (200, 201):
                return True, r.json()
            # 4xx → error del cliente, no reintentar
            if 400 <= r.status_code < 500:
                return False, f"HTTP {r.status_code}: {r.text[:300]}"
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(1.5 * intento)  # backoff
    return False, f"Falló tras {max_retries} intentos: {last_err}"


def main():
    parser = argparse.ArgumentParser(description='Carga familias sintéticas a la BD vía captura-completa')
    parser.add_argument('--input', type=str, default=None,
                        help='families_full.json (default: ../data/families_full.json)')
    parser.add_argument('--base-url', type=str, default='https://api-sums.troy.engineer/sums',
                        help='Base URL del API (incluye /sums)')
    parser.add_argument('--limit', type=int, default=None, help='Cargar solo N familias (prueba)')
    parser.add_argument('--start', type=int, default=1, help='Índice inicial (1-based)')
    parser.add_argument('--delay', type=float, default=0.05, help='Pausa entre requests (seg)')
    parser.add_argument('--timeout', type=float, default=30.0, help='Timeout por request (seg)')
    parser.add_argument('--log', type=str, default=None,
                        help='Log JSONL (default: ../data/processed/load_results.jsonl)')
    parser.add_argument('--user', type=str, default=None, help='Usuario para login (opcional)')
    parser.add_argument('--password', type=str, default=None, help='Contraseña para login (opcional)')
    parser.add_argument('--token', type=str, default=None, help='JWT directo (opcional)')
    parser.add_argument('--no-resume', action='store_true', help='No omitir las ya cargadas')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    input_path = Path(args.input) if args.input else (base_dir / 'data' / 'families_full.json')
    log_path = Path(args.log) if args.log else (base_dir / 'data' / 'processed' / 'load_results.jsonl')
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        sys.exit(f"No existe {input_path}. Corre primero: python synthetic_generator.py")

    familias = cargar_familias(input_path)
    session = requests.Session()
    session.headers.update({'Content-Type': 'application/json'})

    # Conectividad
    print(f"Verificando conexión con {args.base_url} ...")
    if not ping(session, args.base_url, args.timeout):
        print("  ADVERTENCIA: /ping no respondió 200. Continúo de todos modos "
              "(puede que /ping no exista en este despliegue).")
    else:
        print("  OK (pong).")

    # Auth opcional
    token = args.token
    if not token and args.user and args.password:
        print("Autenticando ...")
        token = login(session, args.base_url, args.user, args.password, args.timeout)
        print("  Token obtenido.")
    if token:
        session.headers.update({'Authorization': f'Bearer {token}'})

    # Reanudación
    done = set() if args.no_resume else cargar_indices_hechos(log_path)
    if done:
        print(f"Reanudando: {len(done)} familias ya cargadas se omitirán.")

    # Rango a procesar
    inicio = args.start - 1
    fin = len(familias) if args.limit is None else min(len(familias), inicio + args.limit)

    ok_count = 0
    err_count = 0
    skip_count = 0
    t0 = time.time()

    with open(log_path, 'a', encoding='utf-8') as logf:
        for i in range(inicio, fin):
            idx = i + 1
            if idx in done:
                skip_count += 1
                continue

            payload = familias[i]
            ok, data = post_familia(session, args.base_url, payload, args.timeout)

            if ok:
                ok_count += 1
                rec = {
                    'idx': idx,
                    'ok': True,
                    'cedula_id': data.get('cedula_id'),
                    'nucleo_familiar_id': data.get('nucleo_familiar_id'),
                    'vivienda_id': data.get('vivienda_id'),
                    'n_integrantes': len(data.get('integrantes', [])),
                    'n_inmunizaciones': len(data.get('inmunizaciones', [])),
                    'warnings': data.get('warnings', []),
                }
            else:
                err_count += 1
                rec = {'idx': idx, 'ok': False, 'error': data}

            logf.write(json.dumps(rec, ensure_ascii=False) + '\n')
            logf.flush()

            if idx % 50 == 0 or not ok:
                estado = 'OK ' if ok else 'ERR'
                print(f"  [{idx:5d}/{fin}] {estado}  ok={ok_count} err={err_count}")

            if args.delay:
                time.sleep(args.delay)

    dt = time.time() - t0
    print('=' * 60)
    print(f"  Cargadas OK : {ok_count}")
    print(f"  Errores     : {err_count}")
    print(f"  Omitidas    : {skip_count} (ya estaban)")
    print(f"  Tiempo      : {dt:.1f}s")
    print(f"  Log         : {log_path}")
    print('=' * 60)
    if err_count:
        print("  Revisa el log para ver los errores. Puedes re-correr el script:")
        print("  las exitosas se omiten automáticamente (reanudable).")


if __name__ == '__main__':
    main()
