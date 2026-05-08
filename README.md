日志：
{
  "showThinkingSummaries": false
}


2. 使用流式 JSON 输出并过滤内容
如果你想在实时输出中主动过滤掉 thinking 相关的内容，可以使用 --output-format stream-json，然后通过 jq 只提取最终的文字块：

bash
claude -p "你的问题" --output-format stream-json --verbose --include-partial-messages | \
  jq -rj 'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text'



```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: batch_entry.py",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/batch_entry.py",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "justMyCode": false,
      "args": []
    },
    {
      "name": "Python: batch_entry.py (--stop-on-error)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/batch_entry.py",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "justMyCode": false,
      "args": [
        "--stop-on-error"
      ]
    },
    {
      "name": "Python: concurrent_batch_entry.py",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/concurrent_batch_entry.py",
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "justMyCode": false,
      "args": []
    }
  ]
}
```
