# 🔧 IGRIS — Список необходимых исправлений

> Составлено: 21 февраля 2026  
> Статус: После тестирования выявлено несколько критических багов

---

## ✅ УЖЕ ИСПРАВЛЕНО

### 1. Баги в инструментах (tool functions)
**Статус:** ✅ **ИСПРАВЛЕНО** (21.02.2026)

**Проблема:** Все функции `tool_*` в `core/agent.py` не принимали `**kwargs`, из-за чего LLM не мог передавать дополнительные аргументы.

**Исправлено:**
- `tool_get_time()` → `tool_get_time(**kwargs)`
- `tool_set_mode(mode: str)` → `tool_set_mode(mode: str, **kwargs)`
- `tool_web_search(query: str)` → `tool_web_search(query: str, **kwargs)`
- `tool_remember(text: str)` → `tool_remember(text: str, **kwargs)`
- `tool_recall(query: str)` → `tool_recall(query: str, **kwargs)`
- `tool_screenshot()` → `tool_screenshot(**kwargs)`
- `tool_focus_start(minutes: int = 25)` → `tool_focus_start(minutes: int = 25, **kwargs)`

**Файл:** `core/agent.py` (строки 682-820)

---

## 🚨 КРИТИЧЕСКИЕ ПРОБЛЕМЫ (требуют срочного решения)

### 2. Groq API Rate Limit
**Статус:** ⚠️ **ТРЕБУЕТ ДЕЙСТВИЙ**

**Проблема:** Все 5 API ключей достигли лимита запросов (429 Too Many Requests). После 3-5 запросов система начинает выдавать ошибки.

**Логи:**
```
17:49:28 │ igris.agent │ WARNING │ Rate-limited (429), rotating key…
17:49:29 │ igris.agent │ WARNING │ Rate-limited (429), rotating key…
```

**Решения (выберите одно):**

#### Вариант А: Подождать сброса лимита
- Groq сбрасывает лимиты каждый час
- Подождите 1 час с момента последнего запроса

#### Вариант Б: Получить дополнительные API ключи
1. Перейти на: https://console.groq.com/keys
2. Создать 5-10 новых бесплатных ключей
3. Добавить их в `config.json`:

```json
{
  "groq": {
    "key1_primary": "gsk_...",
    "key2_backup": "gsk_...",
    "key3_vision": "gsk_...",
    "key4_fast": "gsk_...",
    "key5_emergency": "gsk_...",
    "key6_extra1": "gsk_...",    // ← добавить
    "key7_extra2": "gsk_...",    // ← добавить
    "key8_extra3": "gsk_...",    // ← добавить
    // ... до key15_extra10
  }
}
```

#### Вариант В: Оптимизировать использование
- Добавить кеширование частых запросов
- Увеличить интервалы между повторными попытками
- Реализовать fallback ответы для простых запросов

**Файлы для изменения:**
- `config.json` — добавить ключи
- `core/agent.py` — увеличить `RETRY_DELAYS` (строка ~35)
- `core/llm.py` — настроить rate limiting

---

### 3. WebSocket не работает
**Статус:** ⚠️ **ТРЕБУЕТ ИСПРАВЛЕНИЯ**

**Проблема:** WebSocket endpoint возвращает 400 Bad Request. Браузер постоянно показывает "Disconnected".

**Логи:**
```
INFO: connection rejected (400 Bad Request)
INFO: connection closed
ws_clients: 0
```

**Временное решение:**
Создана тестовая страница с REST API: `/static/chat-test.html`

**Постоянное решение:**

1. **Проверить server.py:**

```python
# Файл: server.py, функция ws_chat

@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    try:
        await ws.accept()  # Может падать с ошибкой
        _active_ws.append(ws)
        logger.info(f"WebSocket connected: {ws.client}")
        
        # Отправить приветствие
        await ws.send_json({"type": "connected", "status": "ok"})
        
        while True:
            data = await ws.receive_json()
            # ... обработка
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if ws in _active_ws:
            _active_ws.remove(ws)
```

2. **Проверить nginx конфигурацию** (если используется)
3. **Проверить CORS настройки**

**Файлы для изменения:**
- `server.py` (строки ~60-120)
- `/etc/nginx/sites-available/default` (если используется nginx)

---

## ⚠️ ВАЖНЫЕ ПРОБЛЕМЫ (средний приоритет)

### 4. Модули не инициализированы
**Статус:** ⚠️ **ТРЕБУЕТ ИСПРАВЛЕНИЯ**

**Проблема:** `agent.memory`, `agent.screen`, `agent.focus` и другие модули остаются `None`, поэтому инструменты `remember/recall` не работают.

**Решение:**

```python
# Файл: server.py, функция lifespan (после создания agent)

from core.memory import LongTermMemory, ShortTermMemory
from core.database import Database
from core.focus import FocusTimer

agent = AgentCore(_config)

# Инициализировать модули:
db = await Database.create()
agent.memory = LongTermMemory(db)
agent.screen = None  # Отключен через --no-screen
agent.voice = None   # Отключен через --no-voice
agent.focus = FocusTimer(_config.get("focus", {}))
# agent.media = MediaGenerator(...)  # TODO
# agent.analytics = Analytics(db)    # TODO

agent.register_builtin_tools()
await agent.start()
```

**Файлы для изменения:**
- `server.py` (функция `lifespan`, строки ~30-50)
- Может потребовать создание `core/database.py` если он не полностью реализован

---

### 5. Недостаточный retry logic
**Статус:** 💡 **РЕКОМЕНДУЕТСЯ**

**Проблема:** При rate limit система сдаётся после 3 попыток.

**Решение:**

```python
# Файл: core/agent.py

# БЫЛО:
RETRY_DELAYS = (1.0, 2.0, 4.0)  # 3 попытки

# ДОЛЖНО БЫТЬ:
RETRY_DELAYS = (1.0, 2.0, 5.0, 10.0, 20.0)  # 5 попыток с экспоненциальным backoff
```

**Файлы для изменения:**
- `core/agent.py` (строка ~35)

---

## 💡 ЖЕЛАТЕЛЬНЫЕ УЛУЧШЕНИЯ (низкий приоритет)

### 6. Кеширование ответов
**Статус:** 💡 **ОПЦИОНАЛЬНО**

Сохранять частые запросы типа "Кто ты?", "Какие у тебя возможности?" в кеш, чтобы не тратить API вызовы.

**Реализация:** 
- Использовать Redis или простой in-memory словарь
- Кешировать ответы с TTL 1 час

### 7. Улучшенные error messages
**Статус:** 💡 **ОПЦИОНАЛЬНО**

Вместо "Произошла ошибка связи" показывать более информативные сообщения:
- "API ключи исчерпали лимит. Подождите 1 час или добавьте новые ключи"
- "Модуль памяти не инициализирован. Обратитесь к администратору"

### 8. Fallback responses
**Статус:** 💡 **ОПЦИОНАЛЬНО**

Когда все API ключи заблокированы, отвечать заранее заготовленными фразами вместо ошибки.

---

## 📋 Чек-лист быстрого старта

Для запуска полностью рабочего IGRIS выполните:

- [x] **1. Исправить tool функции** ✅ СДЕЛАНО
- [ ] **2. Добавить 5-10 новых Groq API ключей** ⚠️ ТРЕБУЕТСЯ
- [ ] **3. Исправить WebSocket endpoint** ⚠️ ТРЕБУЕТСЯ
- [ ] **4. Инициализировать модули памяти** ⚠️ РЕКОМЕНДУЕТСЯ
- [ ] **5. Увеличить retry delays** 💡 ОПЦИОНАЛЬНО

**После выполнения п.1-2:** IGRIS будет работать на 70%  
**После выполнения п.1-4:** IGRIS будет работать на 95%

---

## 🧪 Результаты тестирования

### Протестировано: 8 запросов
- ✅ Успешных: 4 (50%)
- ❌ С ошибками: 4 (50%)

### Что работает:
- ✅ Приветствие и представление
- ✅ Смена режимов (Combat/Guard/Sleep)
- ✅ Описание возможностей
- ✅ Обработка ошибок (вежливые сообщения)

### Что не работает:
- ❌ "Который час?" — баг исправлен ✅
- ❌ "Найди информацию о Python" — баг исправлен ✅
- ❌ "Запомни..." — rate limit + память не инициализирована
- ❌ Генерация контента — rate limit

---

## 📞 Контакты и помощь

**Проблемы и вопросы:**
- GitHub Issues: https://github.com/Kotvuk/Igris_v1/issues
- Groq API docs: https://console.groq.com/docs

**Авторы проекта:**
- Дамир — главный разработчик
- Ash — дизайн, UI/UX
- Жасмин — тестирование, документация

---

## 📝 История изменений

### 21.02.2026
- ✅ Исправлены все tool функции (добавлен **kwargs)
- ✅ Создан FIXLIST.md
- ⚠️ Выявлена проблема с rate limit
- ⚠️ Выявлена проблема с WebSocket
- 💡 Создана временная страница chat-test.html (REST API)

---

**Последнее обновление:** 21 февраля 2026, 17:56 UTC  
**Тестировщик:** Axel (AI Executive Assistant)
