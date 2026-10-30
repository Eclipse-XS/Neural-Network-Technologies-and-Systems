$ErrorActionPreference = 'Stop'

$models = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/models' -Method Get
$models | ConvertTo-Json -Depth 8

$body = @{
    model = 'Qwen/Qwen2.5-1.5B-Instruct-AWQ'
    messages = @(
        @{ role = 'system'; content = 'You are a concise assistant.' }
        @{ role = 'user'; content = 'Reply with exactly one short sentence explaining what vLLM does.' }
    )
    max_tokens = 64
    temperature = 0.2
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
    -Uri 'http://127.0.0.1:8000/v1/chat/completions' `
    -Method Post `
    -ContentType 'application/json' `
    -Body $body | ConvertTo-Json -Depth 12
