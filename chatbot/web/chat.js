/*
 * Collegesaurus AI chat page. Plain JavaScript, no build step.
 *
 * URL parameters (the site's chat bubble passes them to the iframe):
 *   embed=true  hide the page header (the bubble panel has its own)
 *   lang=en|ar|fr, page=/universities/aub, theme=light|dark
 *
 * The conversation lives in this tab's sessionStorage, so closing and
 * reopening the bubble (which removes the iframe) keeps it. Answers stream
 * from POST /api/chat as server-sent events: meta, status, delta, discard,
 * sources, error, done.
 */
(() => {
  "use strict";

  const MAX_CHARS = 1000;
  const MAX_HISTORY = 20;
  const MAX_ANSWER_CHARS = 4000;
  const STORE_KEY = "collegesaurus-chat";

  const STRINGS = {
    en: {
      tagline: "Answers from the Collegesaurus guide to Lebanese universities and scholarships.",
      site: "Open the site",
      language: "Language",
      newChat: "New chat",
      welcomeTitle: "What do you want to know?",
      welcomeBody:
        "I answer from Collegesaurus pages on Lebanese universities, majors and scholarships, and link you to the source.",
      suggestions: [
        "What does AUB charge per credit?",
        "Which scholarships pay for study abroad?",
        "What engineering majors does LAU offer?",
        "Which universities do you cover?",
      ],
      placeholder: "Ask about a university, major or scholarship",
      send: "Send",
      privacy:
        "Questions are saved anonymously to improve Collegesaurus. Please don't include personal details.",
      searching: "Searching Collegesaurus",
      searchingUniversities: "Searching universities",
      searchingScholarships: "Searching scholarships",
      listing: "Listing pages",
      thinking: "Thinking",
      sources: "Sources",
      helpful: "Helpful",
      notHelpful: "Not helpful",
      thanks: "Thanks for the feedback",
      retry: "Try again",
      network: "Couldn't reach the assistant. Check your connection and try again.",
      tooLong: "Keep your question under {max} characters.",
    },
    ar: {
      tagline: "إجابات من دليل كوليجسورس للجامعات والمنح في لبنان.",
      site: "افتح الموقع",
      language: "اللغة",
      newChat: "محادثة جديدة",
      welcomeTitle: "ماذا تريد أن تعرف؟",
      welcomeBody:
        "أجيب من صفحات كوليجسورس عن الجامعات اللبنانية والتخصصات والمنح، مع رابط إلى المصدر.",
      suggestions: [
        "كم تبلغ كلفة الساعة المعتمدة في AUB؟",
        "ما المنح التي تموّل الدراسة في الخارج؟",
        "ما تخصصات الهندسة في LAU؟",
        "ما الجامعات التي تغطيها؟",
      ],
      placeholder: "اسأل عن جامعة أو تخصص أو منحة",
      send: "إرسال",
      privacy: "تُحفظ الأسئلة من دون ما يعرّف بك لتحسين كوليجسورس. لا تكتب معلومات شخصية.",
      searching: "أبحث في كوليجسورس",
      searchingUniversities: "أبحث في الجامعات",
      searchingScholarships: "أبحث في المنح",
      listing: "أعدّ قائمة الصفحات",
      thinking: "أفكّر",
      sources: "المصادر",
      helpful: "مفيد",
      notHelpful: "غير مفيد",
      thanks: "شكرًا على ملاحظتك",
      retry: "حاول مجددًا",
      network: "تعذّر الوصول إلى المساعد. تحقّق من اتصالك وحاول مجددًا.",
      tooLong: "أبقِ سؤالك ضمن {max} حرف.",
    },
    fr: {
      tagline: "Des réponses tirées du guide Collegesaurus des universités et bourses au Liban.",
      site: "Ouvrir le site",
      language: "Langue",
      newChat: "Nouvelle discussion",
      welcomeTitle: "Que voulez-vous savoir ?",
      welcomeBody:
        "Je réponds à partir des pages Collegesaurus sur les universités libanaises, les spécialités et les bourses, avec un lien vers la source.",
      suggestions: [
        "Combien coûte un crédit à l'AUB ?",
        "Quelles bourses financent des études à l'étranger ?",
        "Quelles spécialités d'ingénierie propose la LAU ?",
        "Quelles universités couvrez-vous ?",
      ],
      placeholder: "Posez une question sur une université, une spécialité ou une bourse",
      send: "Envoyer",
      privacy:
        "Les questions sont enregistrées anonymement pour améliorer Collegesaurus. N'indiquez pas d'informations personnelles.",
      searching: "Recherche dans Collegesaurus",
      searchingUniversities: "Recherche dans les universités",
      searchingScholarships: "Recherche dans les bourses",
      listing: "Liste des pages",
      thinking: "Réflexion",
      sources: "Sources",
      helpful: "Utile",
      notHelpful: "Pas utile",
      thanks: "Merci pour votre retour",
      retry: "Réessayer",
      network: "Impossible de joindre l'assistant. Vérifiez votre connexion et réessayez.",
      tooLong: "Restez sous {max} caractères.",
    },
  };

  const ICONS = {
    page: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 1.5h5.5L13 5v9.5H4zM9.5 1.5V5H13"/></svg>',
    up: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M6.5 9 9.8 2.8c1 .2 1.7 1 1.6 2.1L11 8h4.4c1 0 1.7.9 1.5 1.9l-1.2 5.6c-.2.8-.9 1.5-1.8 1.5H6.5zM3 9h3.5v8H3z"/></svg>',
    down: '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M13.5 11 10.2 17.2c-1-.2-1.7-1-1.6-2.1L9 12H4.6c-1 0-1.7-.9-1.5-1.9l1.2-5.6C4.5 3.7 5.2 3 6.1 3h7.4zM17 11h-3.5V3H17z"/></svg>',
  };

  const params = new URLSearchParams(location.search);
  const embedded = ["1", "true"].includes(params.get("embed"));
  const page = params.get("page") || "";
  const theme = params.get("theme");

  const $ = (selector) => document.querySelector(selector);
  const els = {
    welcome: $("#welcome"),
    suggestions: $("#suggestions"),
    messages: $("#messages"),
    conversation: $("#conversation"),
    form: $("#composer"),
    input: $("#question"),
    send: $("#send"),
    counter: $("#counter"),
    newChat: $("#new-chat"),
  };

  const saved = load();
  const state = {
    lang: pickLang(params.get("lang") || (saved && saved.lang) || navigator.language),
    sessionId: (saved && saved.sessionId) || newId(),
    messages: (saved && saved.messages) || [],
    busy: false,
    controller: null,
  };

  function pickLang(value) {
    const code = String(value || "").toLowerCase();
    if (code.startsWith("ar")) return "ar";
    if (code.startsWith("fr")) return "fr";
    return "en";
  }

  function t(key) {
    return (STRINGS[state.lang][key] ?? STRINGS.en[key]).toString().replace("{max}", MAX_CHARS);
  }

  function newId() {
    const bytes = new Uint8Array(8);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  }

  // sessionStorage can be unavailable (blocked storage, private modes); the
  // chat still works, it just won't survive the iframe being closed.
  function load() {
    try {
      return JSON.parse(sessionStorage.getItem(STORE_KEY) || "null");
    } catch {
      return null;
    }
  }

  function save() {
    try {
      const messages = state.messages
        .filter((m) => !m.pending)
        .map(({ status, pending, ...kept }) => kept);
      sessionStorage.setItem(
        STORE_KEY,
        JSON.stringify({ lang: state.lang, sessionId: state.sessionId, messages }),
      );
    } catch {
      /* keep going without persistence */
    }
  }

  // ------------------------------------------------------------ language

  function applyLang() {
    const root = document.documentElement;
    root.lang = state.lang;
    root.dir = state.lang === "ar" ? "rtl" : "ltr";
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      el.textContent = t(el.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      el.placeholder = t(el.dataset.i18nPlaceholder);
    });
    document.querySelectorAll("[data-i18n-label]").forEach((el) => {
      el.setAttribute("aria-label", t(el.dataset.i18nLabel));
    });
    document.querySelectorAll(".langs button").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.lang === state.lang));
    });
    els.suggestions.replaceChildren(
      ...STRINGS[state.lang].suggestions.map((text) => {
        const li = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = text;
        button.addEventListener("click", () => send(text));
        li.append(button);
        return li;
      }),
    );
    render();
  }

  // ------------------------------------------------------------ rendering

  function renderMarkdown(text) {
    const html = window.DOMPurify.sanitize(window.marked.parse(text, { gfm: true }), {
      FORBID_ATTR: ["style"],
    });
    const template = document.createElement("template");
    template.innerHTML = html;
    template.content.querySelectorAll("table").forEach((table) => {
      const wrap = document.createElement("div");
      wrap.className = "table-wrap";
      table.replaceWith(wrap);
      wrap.append(table);
    });
    template.content.querySelectorAll("a[href]").forEach((a) => {
      let url;
      try {
        url = new URL(a.getAttribute("href"), location.href);
      } catch {
        return;
      }
      const onSite = url.hostname === "collegesaurus.org" || url.hostname.endsWith(".collegesaurus.org");
      // Inside the site's bubble, site links open in the page itself.
      a.target = onSite && embedded ? "_top" : "_blank";
      a.rel = "noopener noreferrer";
    });
    return template.content;
  }

  function statusText(data) {
    if (data.tool === "list_pages") return t("listing");
    if (data.type === "university") return t("searchingUniversities");
    if (data.type === "scholarship") return t("searchingScholarships");
    return t("searching");
  }

  function renderMessage(message, index) {
    const li = document.createElement("li");
    if (message.role === "user") {
      li.className = "msg-user";
      li.dir = "auto";
      li.textContent = message.content;
      return li;
    }
    li.className = "msg-bot";
    li.dir = "auto";
    li.setAttribute("aria-busy", String(Boolean(message.pending)));

    const answer = document.createElement("div");
    answer.className = "answer";
    if (message.content) answer.append(renderMarkdown(message.content));
    li.append(answer);

    if (message.pending && !message.content) {
      const status = document.createElement("div");
      status.className = "status";
      status.innerHTML = '<span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>';
      status.append(document.createTextNode(message.status || t("thinking")));
      li.append(status);
    }

    if (message.error) {
      const notice = document.createElement("div");
      notice.className = "notice";
      notice.setAttribute("role", "alert");
      notice.append(document.createTextNode(message.error));
      if (index === state.messages.length - 1) {
        const retry = document.createElement("button");
        retry.type = "button";
        retry.textContent = t("retry");
        retry.addEventListener("click", retryLast);
        notice.append(retry);
      }
      li.append(notice);
    }

    const canRate = !message.pending && message.turnId && message.outcome === "answered";
    if ((message.sources && message.sources.length) || canRate) {
      const foot = document.createElement("div");
      foot.className = "msg-foot";
      if (message.sources && message.sources.length) {
        const sources = document.createElement("div");
        sources.className = "sources";
        const label = document.createElement("span");
        label.className = "sources-label";
        label.textContent = t("sources");
        sources.append(label);
        for (const source of message.sources) {
          const a = document.createElement("a");
          a.className = "source";
          a.href = source.url;
          a.title = source.title;
          a.target = embedded ? "_top" : "_blank";
          a.rel = "noopener noreferrer";
          a.innerHTML = ICONS.page;
          const name = document.createElement("span");
          name.textContent = source.title.split(" — ")[0];
          a.append(name);
          sources.append(a);
        }
        foot.append(sources);
      }
      if (canRate) foot.append(renderFeedback(message));
      li.append(foot);
    }
    return li;
  }

  function renderFeedback(message) {
    const box = document.createElement("div");
    box.className = "feedback";
    if (message.feedback) {
      const thanks = document.createElement("span");
      thanks.textContent = t("thanks");
      box.append(thanks);
    }
    for (const [value, icon, label] of [
      [1, ICONS.up, "helpful"],
      [-1, ICONS.down, "notHelpful"],
    ]) {
      const button = document.createElement("button");
      button.type = "button";
      button.innerHTML = icon;
      button.setAttribute("aria-label", t(label));
      button.title = t(label);
      button.setAttribute("aria-pressed", String(message.feedback === value));
      button.disabled = Boolean(message.feedback);
      button.addEventListener("click", () => rate(message, value));
      box.append(button);
    }
    return box;
  }

  function render() {
    els.welcome.hidden = state.messages.length > 0;
    els.messages.replaceChildren(...state.messages.map(renderMessage));
    els.send.disabled = state.busy || els.input.value.trim().length > MAX_CHARS;
  }

  // Re-render only the streaming message, at most once per frame.
  let frame = 0;
  function refreshLast() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      const stick = nearBottom();
      const index = state.messages.length - 1;
      const node = els.messages.lastElementChild;
      if (index < 0 || !node) return;
      node.replaceWith(renderMessage(state.messages[index], index));
      if (stick) scrollToEnd();
    });
  }

  function nearBottom() {
    const c = els.conversation;
    return c.scrollHeight - c.scrollTop - c.clientHeight < 80;
  }

  function scrollToEnd() {
    els.conversation.scrollTop = els.conversation.scrollHeight;
  }

  // ------------------------------------------------------------ talking to the server

  function history() {
    return state.messages
      .filter((m) => !m.pending && !m.error && m.content)
      .slice(-MAX_HISTORY)
      .map((m) => ({
        role: m.role,
        content: m.role === "assistant" ? m.content.slice(0, MAX_ANSWER_CHARS) : m.content,
      }));
  }

  async function send(question) {
    const text = question.trim();
    if (!text || state.busy) return;
    if (text.length > MAX_CHARS) {
      updateCounter();
      return;
    }
    state.messages.push({ role: "user", content: text });
    const bot = { role: "assistant", content: "", pending: true, status: "", sources: [] };
    state.messages.push(bot);
    els.input.value = "";
    autosize();
    updateCounter();
    state.busy = true;
    render();
    scrollToEnd();
    save();

    const controller = new AbortController();
    state.controller = controller;
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history(),
          lang: state.lang,
          page,
          session_id: state.sessionId,
        }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => ({}));
        bot.error = data.message || t("network");
      } else {
        await readEvents(response.body, (event, data) => handleEvent(bot, event, data));
        if (!bot.outcome && !bot.error) bot.error = t("network");
      }
    } catch (error) {
      if (error.name === "AbortError") return;
      bot.error = t("network");
    } finally {
      if (state.controller === controller) {
        state.controller = null;
        state.busy = false;
        bot.pending = false;
        render();
        if (nearBottom()) scrollToEnd();
        save();
      }
    }
  }

  function handleEvent(bot, event, data) {
    switch (event) {
      case "meta":
        bot.turnId = data.turn_id;
        break;
      case "status":
        bot.status = statusText(data);
        break;
      case "delta":
        bot.content += data.text;
        break;
      case "discard":
        bot.content = "";
        break;
      case "sources":
        bot.sources = data.items;
        break;
      case "error":
        bot.error = data.message;
        break;
      case "done":
        bot.outcome = data.outcome;
        break;
    }
    refreshLast();
  }

  async function readEvents(body, onEvent) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        let event = "message";
        const data = [];
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) data.push(line.slice(6));
        }
        if (data.length) onEvent(event, JSON.parse(data.join("\n")));
      }
    }
  }

  function retryLast() {
    const last = state.messages[state.messages.length - 1];
    const question = state.messages[state.messages.length - 2];
    if (!last || last.role !== "assistant" || !question || question.role !== "user") return;
    state.messages.splice(-2, 2);
    send(question.content);
  }

  function rate(message, value) {
    if (message.feedback || !message.turnId) return;
    message.feedback = value;
    render();
    save();
    fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ turn_id: message.turnId, value }),
    }).catch(() => {});
  }

  function newChat() {
    if (state.controller) state.controller.abort();
    state.controller = null;
    state.busy = false;
    state.messages = [];
    state.sessionId = newId();
    render();
    save();
    els.input.focus();
  }

  // ------------------------------------------------------------ composer

  function autosize() {
    els.input.style.height = "auto";
    els.input.style.height = `${Math.min(els.input.scrollHeight + 2, 144)}px`;
  }

  function updateCounter() {
    const length = els.input.value.trim().length;
    els.counter.hidden = length < MAX_CHARS * 0.8;
    els.counter.textContent = `${length}/${MAX_CHARS}`;
    els.counter.classList.toggle("over", length > MAX_CHARS);
    els.counter.title = length > MAX_CHARS ? t("tooLong") : "";
    els.send.disabled = state.busy || length > MAX_CHARS;
  }

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    send(els.input.value);
  });
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send(els.input.value);
    }
  });
  els.input.addEventListener("input", () => {
    autosize();
    updateCounter();
  });
  els.newChat.addEventListener("click", newChat);
  document.querySelectorAll(".langs button").forEach((button) => {
    button.addEventListener("click", () => {
      state.lang = button.dataset.lang;
      applyLang();
      save();
    });
  });

  if (embedded) document.documentElement.classList.add("is-embed");
  if (theme === "dark" || theme === "light") document.documentElement.dataset.theme = theme;
  applyLang();
  scrollToEnd();
})();
