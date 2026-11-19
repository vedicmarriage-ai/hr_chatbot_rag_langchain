# Include inside each SPA server block that hosts the chatbot widget
# (www.vedicmarriage.com, staging.vedicmarriage.com, vedicmarriage.in).
# Must sit before `location /` try_files so POST /hr-chat/api/ask is not 405.

location /hr-chat/ {
    proxy_pass http://127.0.0.1:8010/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 90s;
    proxy_send_timeout 90s;
    client_max_body_size 32k;
}
