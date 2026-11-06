// Supabase Edge Function：代理富果行情 API，讓前端不必暴露 API key。
// 部署：supabase functions deploy fugle-quote --no-verify-jwt
// 設定金鑰：supabase secrets set FUGLE_API_KEY=你的key
// 呼叫：GET {SUPABASE_URL}/functions/v1/fugle-quote?symbols=2330,2317

const FUGLE = "https://api.fugle.tw/marketdata/v1.0/stock/intraday/quote/";
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  const key = Deno.env.get("FUGLE_API_KEY");
  if (!key) return new Response(JSON.stringify({ error: "FUGLE_API_KEY not set" }), { status: 500, headers: CORS });

  const url = new URL(req.url);
  const symbols = (url.searchParams.get("symbols") ?? "").split(",").map((s) => s.trim()).filter(Boolean).slice(0, 20);
  if (!symbols.length) return new Response(JSON.stringify({ error: "symbols required" }), { status: 400, headers: CORS });

  const results = await Promise.all(symbols.map(async (sym) => {
    try {
      const r = await fetch(FUGLE + sym, { headers: { "X-API-KEY": key } });
      if (!r.ok) return { symbol: sym, error: r.status };
      const j = await r.json();
      return {
        symbol: sym,
        name: j.name,
        price: j.lastPrice ?? j.closePrice ?? null,
        change: j.change ?? null,
        changePercent: j.changePercent ?? null,
        volume: j.total?.tradeVolume ?? null,
        time: j.lastUpdated ?? null,
        isClose: j.isClose ?? null,
      };
    } catch (e) {
      return { symbol: sym, error: String(e) };
    }
  }));

  return new Response(JSON.stringify({ quotes: results, fetchedAt: new Date().toISOString() }), {
    headers: { ...CORS, "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
});
