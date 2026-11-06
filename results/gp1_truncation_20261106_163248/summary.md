# GP1 truncation check, 20261106_163248

Model `openai/gpt-oss-120b`, `max_tokens` 200, 10 sentences through `parse_expense` (§3.1).

| Level | Valid JSON | Cut at max_tokens | Crashes | Reasoning tokens (mean / max) | Largest completion | Tokens |
|---|---|---|---|---|---|---|
| default | 5/10 | 6 | 0 | 147.8 / 198 | 200 | 10687 |
| low | 10/10 | 0 | 0 | 44.6 / 72 | 122 | 9874 |

| Sentence | Expected | default: type, finish, reasoning | low: type, finish, reasoning |
|---|---|---|---|
| دفعت فاتورة الموبايل 5000 | expense | expense, stop, 103 | expense, stop, 29 |
| اشتريت رز وزيت بـ 6000 | expense | expense, stop, 125 | expense, stop, 55 |
| رحت عأسنان دفعت 25 ألف | expense | error, length, 198 | expense, stop, 71 |
| توفير هالشهر 25 ألف | expense | advice, stop, 110 | advice, stop, 17 |
| عبيت بنزين نص تنكة 7000 | expense | error, length, 172 | expense, stop, 45 |
| عشا بره بـ 12000 | expense | error, length, 191 | expense, stop, 47 |
| كورس انجليزي 25 ألف | expense | error, length, 198 | expense, stop, 42 |
| دورة برمجة 40 ألف | expense | expense, length, 150 | expense, stop, 72 |
| اشتراك OSN 10 ألف | expense | error, length, 198 | expense, stop, 55 |
| شو نصيحتك لتقليل المصاريف؟ | advice | advice, stop, 33 | advice, stop, 13 |
