// BuscadorCasos.tsx
// ----------------------------------------------------------------------------
// Vista de EJEMPLO para SUMS_WEB: buscador de notas de visita domiciliaria que
// consume el microservicio de minería (Subcomponente C) en /buscar.
//
// Cómo usarla:
//   1) Levanta la API:  uvicorn api_mineria:app --port 8001
//   2) Copia este archivo a SUMS_WEB (p.ej. src/pages/BuscadorCasos.tsx)
//   3) Agrégalo al router como una página más (ruta /buscador)
//
// Estilos con Tailwind (ya está en el proyecto). Sin dependencias extra.
// ----------------------------------------------------------------------------
import { useState } from "react";

const API_MINERIA = import.meta.env.VITE_MINERIA_URL ?? "http://localhost:8001";

type Resultado = {
  posicion: number;
  id: string;
  titulo: string;
  score: number;
  texto: string;
};

export default function BuscadorCasos() {
  const [q, setQ] = useState("familias con desnutrición infantil");
  const [motor, setMotor] = useState<"bm25" | "tfidf">("bm25");
  const [resultados, setResultados] = useState<Resultado[]>([]);
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function buscar(e?: React.FormEvent) {
    e?.preventDefault();
    if (!q.trim()) return;
    setCargando(true);
    setError(null);
    try {
      const url = `${API_MINERIA}/buscar?q=${encodeURIComponent(q)}&motor=${motor}&k=8`;
      const r = await fetch(url);
      if (!r.ok) throw new Error(`Error ${r.status}`);
      const data = await r.json();
      setResultados(data.resultados ?? []);
    } catch (err) {
      setError((err as Error).message);
      setResultados([]);
    } finally {
      setCargando(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl p-6">
      <h1 className="mb-1 text-2xl font-bold text-slate-800">Buscador de casos</h1>
      <p className="mb-4 text-sm text-slate-500">
        Busca en las notas de visita domiciliaria (motor {motor.toUpperCase()}).
      </p>

      <form onSubmit={buscar} className="mb-6 flex gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="ej. casos sospechosos de dengue en San José"
          className="flex-1 rounded-lg border border-slate-300 px-4 py-2 outline-none focus:border-sky-500"
        />
        <select
          value={motor}
          onChange={(e) => setMotor(e.target.value as "bm25" | "tfidf")}
          className="rounded-lg border border-slate-300 px-2 py-2 text-sm"
          title="Motor de búsqueda"
        >
          <option value="bm25">BM25</option>
          <option value="tfidf">TF-IDF</option>
        </select>
        <button
          type="submit"
          disabled={cargando}
          className="rounded-lg bg-sky-600 px-5 py-2 font-medium text-white hover:bg-sky-700 disabled:opacity-50"
        >
          {cargando ? "Buscando…" : "Buscar"}
        </button>
      </form>

      {error && <p className="mb-4 text-sm text-red-600">⚠ {error}</p>}

      <ul className="space-y-3">
        {resultados.map((r) => (
          <li key={r.id} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-1 flex items-center justify-between">
              <span className="font-semibold text-slate-800">{r.titulo}</span>
              <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
                #{r.posicion} · score {r.score.toFixed(2)} · {r.id}
              </span>
            </div>
            <p className="text-sm text-slate-600">{r.texto}</p>
          </li>
        ))}
      </ul>

      {!cargando && resultados.length === 0 && !error && (
        <p className="text-sm text-slate-400">Sin resultados todavía. Escribe una consulta y busca.</p>
      )}
    </div>
  );
}
