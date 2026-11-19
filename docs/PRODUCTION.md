# Production: widget → this service

The design-library widget posts to **same-origin** `POST /hr-chat/api/ask`. Vite rewrites that path in `npm run dev`. **nginx in product does not**, so the SPA returns **405** on POST (or `index.html` on GET) and the widget shows “The support bot is unavailable.”

## Required

1. This process listening on **`127.0.0.1:8010`** (not published on the public NIC).
2. nginx `location /hr-chat/` that strips the prefix and proxies here — see [`deploy/nginx-hr-chat.inc`](../deploy/nginx-hr-chat.inc). Applied on www, staging, and `.in` SPA blocks.

```bash
curl -sS https://staging.vedicmarriage.com/hr-chat/health
# {"status":"healthy","version":"..."}
```

3. Optional systemd unit: [`deploy/hr-chatbot.service`](../deploy/hr-chatbot.service).

Do not open port 8010 on the firewall. CORS is a fallback only; the product path is same-origin nginx.
