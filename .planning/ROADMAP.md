# ROADMAP

## Phase 1: RLM Core + Basic API
**Goal**: RLM-сервер работает с DeepSeek, отдаёт OpenAI-compatible API

### Plans
**3 plans in 3 waves**

Plans:
- [ ] 01-01-PLAN.md — Project scaffold, dependencies, config module (wave 1)
- [ ] 01-02-PLAN.md — RLM client + system prompt for DeepSeek (wave 2)
- [ ] 01-03-PLAN.md — FastAPI server with /v1/chat/completions endpoint (wave 3)

### Verification
- `curl` запрос к API возвращает ответ от DeepSeek через RLM
- VS Code Copilot может подключиться к endpoint

---

## Phase 2: Multi-Model + Custom Tools
**Goal**: DeepSeek Pro для root, Flash для sub-calls, dev tools в REPL

### Plans
1. **[P2.1] Model routing** — Pro для сложных задач, Flash для sub-calls
2. **[P2.2] Custom tools** — файловые операции, запуск команд, анализ кода
3. **[P2.3] System prompt tuning** — оркестратор для multi-model delegation
4. **[P2.4] Mimo integration** — альтернативный backend

### Verification
- RLM автоматически выбирает модель по сложности задачи
- Custom tools работают внутри REPL (чтение файлов, запуск pytest)

---

## Phase 3: RAG для кодовой базы
**Goal**: RLM индексирует и ищет по исходникам проекта

### Plans
1. **[P3.1] ChromaDB setup** — векторная БД для кода
2. **[P3.2] Code indexer** — парсинг .py/.ts/.rs файлов, chunking
3. **[P3.3] RAG tool** — custom tool для поиска по кодовой базе
4. **[P3.4] Context injection** — автоматическое добавление релевантного контекста

### Verification
- Вопрос "где используется функция X?" находит правильные файлы
- RLM автоматически подтягивает контекст из кодовой базы

---

## Phase 4: Production + Optimization
**Goal**: Стабильный, быстрый, готовый к ежедневному использованию

### Plans
1. **[P4.1] Streaming** — SSE для стриминга ответов
2. **[P4.2] Caching** — кеширование повторных запросов
3. **[P4.3] Error handling** — retry, fallback между моделями
4. **[P4.4] Logging + monitoring** — трекинг стоимости, latency

### Verification
- Стриминг работает в VS Code Copilot
- При ошибке DeepSeek автоматически переключается на Mimo
- Видна статистика по токенам и стоимости

---

## Dependencies

```
Phase 1 → Phase 2 (нужен работающий RLM ядро)
Phase 2 → Phase 3 (нужны custom tools для RAG)
Phase 1 → Phase 4 (нужен базовый API для streaming/caching)
Phase 3 → Phase 4 (RAG должен работать до оптимизации)
```

## Success Criteria

- [ ] VS Code Copilot использует RLM для ответов на вопросы о коде
- [ ] RLM рекурсивно обрабатывает длинные файлы (>100K tokens)
- [ ] Стоимость ниже прямого использования DeepSeek Pro для тех же задач
- [ ] Latency < 10s для типичных dev-запросов

---
*Created: 2026-06-14*
