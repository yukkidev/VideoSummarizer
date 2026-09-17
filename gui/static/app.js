"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  currentVideo: null,
  videoId: null,
  scope: "video",
  threads: [],
  threadId: null,
  currentThread: null,
  playlists: [],
  videoList: [],
  expandedPlaylists: new Set(),
  context: null,
  mode: "ask",
  jobTimer: null,
  models: null,
};

// ---------- utilities -------------------------------------------------------

function fmtTime(seconds) {
  seconds = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function escapeHtml(text) {
  return (text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function timestampToSeconds(match) {
  const parts = match.split(":").map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return 0;
}

function linkifyTimestamps(html) {
  return html
    .replace(/\[(\d{1,2}:\d{2})-(\d{1,2}:\d{2}(?::\d{2})?)\]/g, (_, start) =>
      `<span class="ts-link" data-seconds="${timestampToSeconds(start)}">[${start}]</span>`)
    .replace(/\[(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})\]/g, (_, stamp) =>
      `<span class="ts-link" data-seconds="${timestampToSeconds(stamp)}">[${stamp}]</span>`);
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await response.text();
  let payload = {};
  try { payload = text ? JSON.parse(text) : {}; } catch { payload = { raw: text }; }
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function seekTo(seconds) {
  const player = $("player");
  if (!player || !player.src) return;
  player.currentTime = seconds;
  player.play().catch(() => {});
  player.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------- notifications ---------------------------------------------------

function requestNotifyPermission() {
  if (!("Notification" in window)) return;
  if (Notification.permission === "default") {
    Notification.requestPermission().catch(() => {});
  }
}

function notifyFinished(job) {
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  const result = (job && job.result) || {};
  let body = "Processing finished";
  if (job && job.kind === "pull") {
    body = `Finished pulling ${result.model || "model"}`;
  } else if (result.title) {
    body = `Finished processing: ${result.title}`;
  } else if (result.video_id) {
    body = `Finished processing ${result.video_id}`;
  }
  try {
    new Notification("VideoSummarizer", { body });
  } catch {
    // notifications are best-effort
  }
}

// ---------- playlists -------------------------------------------------------

function videoTitle(videoId) {
  const video = state.videoList.find((v) => v.video_id === videoId);
  return video ? (video.title || videoId) : videoId;
}

async function loadPlaylists() {
  try {
    state.playlists = await apiFetch("/api/playlists");
  } catch {
    state.playlists = [];
  }
  renderPlaylists();
  renderScope();
}

function renderPlaylists() {
  const list = $("playlist-list");
  list.innerHTML = "";
  if (!state.playlists.length) {
    const hint = document.createElement("p");
    hint.className = "muted";
    hint.textContent = "Group related videos to keep chats focused.";
    list.appendChild(hint);
    return;
  }
  state.playlists.forEach((playlist) => {
    const item = document.createElement("div");
    item.className = "playlist";
    const head = document.createElement("div");
    head.className = "playlist-head";

    const expanded = state.expandedPlaylists.has(playlist.playlist_id);
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "playlist-toggle";
    toggle.textContent = `${expanded ? "▾" : "▸"} ${playlist.name} (${playlist.video_ids.length})`;
    toggle.title = "Expand / collapse";
    toggle.addEventListener("click", () => {
      if (expanded) state.expandedPlaylists.delete(playlist.playlist_id);
      else state.expandedPlaylists.add(playlist.playlist_id);
      renderPlaylists();
    });
    head.appendChild(toggle);

    const addCurrent = document.createElement("button");
    addCurrent.type = "button";
    addCurrent.className = "icon-btn small";
    addCurrent.textContent = "＋";
    addCurrent.title = "Add the current video to this playlist";
    addCurrent.disabled = !state.videoId || playlist.video_ids.includes(state.videoId);
    addCurrent.addEventListener("click", () => addVideoToPlaylist(playlist.playlist_id, state.videoId));
    head.appendChild(addCurrent);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-btn small";
    remove.textContent = "×";
    remove.title = "Delete playlist (videos stay in the library)";
    remove.addEventListener("click", () => deletePlaylist(playlist.playlist_id, playlist.name));
    head.appendChild(remove);
    item.appendChild(head);

    if (expanded) {
      const body = document.createElement("div");
      body.className = "playlist-body";

      if (!playlist.video_ids.length) {
        const empty = document.createElement("p");
        empty.className = "muted";
        empty.textContent = "No videos yet.";
        body.appendChild(empty);
      }
      playlist.video_ids.forEach((videoId) => {
        const row = document.createElement("div");
        row.className = "playlist-video";
        const open = document.createElement("button");
        open.type = "button";
        open.className = "playlist-video-open";
        open.textContent = videoTitle(videoId);
        open.title = videoId;
        open.addEventListener("click", () => openVideo(videoId));
        row.appendChild(open);
        const drop = document.createElement("button");
        drop.type = "button";
        drop.className = "icon-btn small";
        drop.textContent = "×";
        drop.title = "Remove from playlist";
        drop.addEventListener("click", () => removeVideoFromPlaylist(playlist.playlist_id, videoId));
        row.appendChild(drop);
        body.appendChild(row);
      });

      const select = document.createElement("select");
      select.className = "playlist-add-select";
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Add video…";
      select.appendChild(placeholder);
      state.videoList
        .filter((video) => !playlist.video_ids.includes(video.video_id))
        .forEach((video) => {
          const option = document.createElement("option");
          option.value = video.video_id;
          option.textContent = video.title || video.video_id;
          select.appendChild(option);
        });
      select.addEventListener("change", () => {
        if (select.value) addVideoToPlaylist(playlist.playlist_id, select.value);
      });
      body.appendChild(select);
      item.appendChild(body);
    }
    list.appendChild(item);
  });
}

async function createPlaylist() {
  const name = prompt("Playlist name (e.g. Quantum mechanics):");
  if (!name || !name.trim()) return;
  try {
    const playlist = await apiFetch("/api/playlists", {
      method: "POST",
      body: JSON.stringify({ name: name.trim() }),
    });
    state.expandedPlaylists.add(playlist.playlist_id);
    await loadPlaylists();
  } catch (error) {
    alert(`Could not create playlist: ${error.message}`);
  }
}

async function deletePlaylist(playlistId, name) {
  if (!confirm(`Delete playlist "${name}"? The videos stay in your library.`)) return;
  await apiFetch(`/api/playlists/${playlistId}`, { method: "DELETE" });
  state.expandedPlaylists.delete(playlistId);
  await loadPlaylists();
}

async function addVideoToPlaylist(playlistId, videoId) {
  if (!videoId) return;
  await apiFetch(`/api/playlists/${playlistId}`, {
    method: "PATCH",
    body: JSON.stringify({ add_video_ids: [videoId] }),
  });
  await loadPlaylists();
}

async function removeVideoFromPlaylist(playlistId, videoId) {
  await apiFetch(`/api/playlists/${playlistId}`, {
    method: "PATCH",
    body: JSON.stringify({ remove_video_ids: [videoId] }),
  });
  await loadPlaylists();
}

// ---------- global chat scope ----------------------------------------------

function scopeSummary() {
  const ids = (state.currentThread && state.currentThread.playlist_ids) || [];
  if (!ids.length) return "all videos";
  const selected = state.playlists.filter((p) => ids.includes(p.playlist_id));
  const videos = new Set();
  selected.forEach((p) => p.video_ids.forEach((v) => videos.add(v)));
  const names = selected.map((p) => p.name);
  const label = names.length <= 2 ? names.join(", ") : `${names.length} playlists`;
  return `${label} · ${videos.size} video${videos.size === 1 ? "" : "s"}`;
}

function renderScope() {
  const picker = $("scope-picker");
  if (!picker) return;
  picker.classList.toggle("hidden", state.scope !== "global");
  $("scope-toggle").textContent = `Scope: ${scopeSummary()} ▾`;

  const ids = (state.currentThread && state.currentThread.playlist_ids) || [];
  const all = $("scope-all");
  all.checked = ids.length === 0;
  const container = $("scope-playlists");
  container.innerHTML = "";
  if (!state.playlists.length) {
    const hint = document.createElement("p");
    hint.className = "muted";
    hint.textContent = "No playlists yet.";
    container.appendChild(hint);
    return;
  }
  state.playlists.forEach((playlist) => {
    const label = document.createElement("label");
    label.className = "scope-option";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = playlist.playlist_id;
    box.checked = ids.includes(playlist.playlist_id);
    box.disabled = all.checked;
    label.appendChild(box);
    const text = document.createElement("span");
    text.textContent = `${playlist.name} (${playlist.video_ids.length})`;
    label.appendChild(text);
    container.appendChild(label);
  });
}

async function applyScope() {
  if (!state.currentThread) return;
  const ids = $("scope-all").checked
    ? []
    : Array.from($("scope-playlists").querySelectorAll("input:checked"))
        .map((box) => box.value);
  try {
    state.currentThread = await apiFetch(`/api/threads/${state.threadId}`, {
      method: "PATCH",
      body: JSON.stringify({ playlist_ids: ids }),
    });
  } catch (error) {
    alert(`Could not update scope: ${error.message}`);
    return;
  }
  $("scope-menu").classList.add("hidden");
  renderScope();
}

// ---------- context window --------------------------------------------------

function fmtTokens(value) {
  return (value || 0).toLocaleString();
}

function contextLimitValue(context) {
  return (context && (context.context_limit || context.loaded_context || context.max_context)) || 0;
}

function contextUsed(context) {
  if (!context) return 0;
  return context.prompt_tokens || context.thread_tokens || 0;
}

function renderContext(context) {
  if (context) state.context = context;
  const ctx = state.context;
  if (!ctx) return;
  const limit = contextLimitValue(ctx);
  const used = contextUsed(ctx);
  const pct = limit ? Math.min(100, Math.round((used / limit) * 100)) : 0;
  $("context-btn").textContent = limit && used ? `Context ${pct}%` : "Context";

  const body = $("context-body");
  body.innerHTML = "";
  const rows = [
    ["Model", ctx.model || "—"],
    [
      "Context loaded by Ollama",
      ctx.loaded_context
        ? `${fmtTokens(ctx.loaded_context)} tokens`
        : "not loaded yet — send a message first",
    ],
    [
      "Model maximum",
      ctx.max_context ? `${fmtTokens(ctx.max_context)} tokens` : "unknown",
    ],
    [
      "Last request",
      ctx.prompt_tokens
        ? `${fmtTokens(ctx.prompt_tokens)} prompt + ${fmtTokens(ctx.eval_tokens)} generated`
        : "no replies yet",
    ],
    [
      "Conversation estimate",
      `~${fmtTokens(ctx.thread_tokens)} tokens · ${ctx.thread_messages} messages`,
    ],
  ];
  rows.forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "context-row";
    const left = document.createElement("span");
    left.className = "context-label";
    left.textContent = label;
    const right = document.createElement("span");
    right.className = "context-value";
    right.textContent = value;
    row.appendChild(left);
    row.appendChild(right);
    body.appendChild(row);
  });

  const bar = document.createElement("div");
  bar.className = "progress context-bar";
  const fill = document.createElement("div");
  fill.className = "progress-fill";
  fill.style.width = `${pct}%`;
  if (pct > 85) fill.style.background = "var(--bad)";
  bar.appendChild(fill);
  body.appendChild(bar);

  const note = document.createElement("p");
  note.className = "muted";
  note.textContent = pct
    ? `${pct}% of the loaded window used by the last request.`
    : "Token counts come from Ollama's last reply plus a 4-chars-per-token estimate.";
  body.appendChild(note);
}

async function refreshContext() {
  try {
    const query = state.threadId ? `?thread_id=${state.threadId}` : "";
    renderContext(await apiFetch(`/api/context${query}`));
  } catch {
    // context display is best-effort
  }
}

// ---------- models ----------------------------------------------------------

async function refreshModels() {
  const btn = $("refresh-models");
  btn.disabled = true;
  try {
    const status = await apiFetch("/api/models");
    state.models = status;
    const select = $("model-select");
    select.innerHTML = "";
    (status.installed || []).forEach((model) => {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = `${model.name}  (${model.size_gb} GB)`;
      if (model.name === status.current) option.selected = true;
      select.appendChild(option);
    });
    if (!(status.installed || []).length) {
      const option = document.createElement("option");
      option.textContent = "no models installed";
      option.value = "";
      select.appendChild(option);
    }
    const dot = $("server-dot");
    dot.className = "dot " + (status.server_ok ? "ok" : "bad");
    dot.title = status.server_ok ? "Ollama online" : (status.error || "Ollama offline");
    const badge = $("loaded-badge");
    if (status.current_loaded) {
      badge.textContent = "loaded in memory";
      badge.className = "badge loaded";
    } else {
      badge.textContent = status.server_ok ? "not loaded" : "offline";
      badge.className = "badge";
    }
  } catch (error) {
    $("server-dot").className = "dot bad";
    $("server-dot").title = error.message;
  } finally {
    btn.disabled = false;
  }
}

async function switchModel(name) {
  if (!name) return;
  try {
    const status = await apiFetch("/api/models/current", {
      method: "POST",
      body: JSON.stringify({ model: name }),
    });
    state.models = status;
    const badge = $("loaded-badge");
    badge.textContent = status.current_loaded ? "loaded in memory" : "not loaded";
    badge.className = "badge" + (status.current_loaded ? " loaded" : "");
  } catch (error) {
    alert(`Could not switch model: ${error.message}`);
  }
}

async function pullModel() {
  const name = prompt("Model to pull (e.g. llama3.2:3b):");
  if (!name) return;
  try {
    const { job_id } = await apiFetch("/api/models/pull", {
      method: "POST",
      body: JSON.stringify({ model: name }),
    });
    showJob(`pulling ${name}`);
    pollJob(job_id, {
      onDone: async (job) => { notifyFinished(job); await refreshModels(); },
      onError: (message) => showJobError(message),
    });
  } catch (error) {
    alert(`Pull failed: ${error.message}`);
  }
}

// ---------- jobs ------------------------------------------------------------

function showJob(message) {
  $("job-box").classList.remove("hidden");
  $("job-error").classList.add("hidden");
  $("job-message").textContent = message;
  $("job-pct").textContent = "0%";
  $("job-bar").style.width = "0%";
}

function showJobError(message) {
  $("job-error").textContent = message;
  $("job-error").classList.remove("hidden");
}

function pollJob(jobId, handlers = {}) {
  clearInterval(state.jobTimer);
  state.jobTimer = setInterval(async () => {
    try {
      const job = await apiFetch(`/api/jobs/${jobId}`);
      $("job-message").textContent = job.message || job.state;
      $("job-pct").textContent = `${Math.round((job.fraction || 0) * 100)}%`;
      $("job-bar").style.width = `${Math.round((job.fraction || 0) * 100)}%`;
      if (job.state === "done") {
        clearInterval(state.jobTimer);
        setTimeout(() => $("job-box").classList.add("hidden"), 900);
        if (handlers.onDone) handlers.onDone(job);
      } else if (job.state === "error") {
        clearInterval(state.jobTimer);
        showJobError(job.error || "job failed");
        if (handlers.onError) handlers.onError(job.error);
      }
    } catch (error) {
      clearInterval(state.jobTimer);
      showJobError(error.message);
    }
  }, 600);
}

// ---------- video library ---------------------------------------------------

function videoFlags(video) {
  return ["has_video", "has_audio", "has_subtitles", "has_transcript", "has_summary", "has_vectors"]
    .filter((key) => video[key])
    .map((key) => key.replace("has_", ""))
    .join(" · ");
}

async function loadVideos() {
  const videos = await apiFetch("/api/videos");
  state.videoList = videos;
  renderPlaylists();
  const list = $("video-list");
  list.innerHTML = "";
  if (!videos.length) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "No videos yet.";
    list.appendChild(li);
    return;
  }
  videos.forEach((video) => {
    const li = document.createElement("li");
    li.dataset.id = video.video_id;
    if (video.video_id === state.videoId) li.classList.add("active");
    li.innerHTML = `
      <span class="v-title">${escapeHtml(video.title || video.video_id)}</span>
      <span class="v-sub">${escapeHtml(video.author || "")} · ${fmtTime(video.duration || 0)}</span>
      <div class="v-flags">${videoFlags(video)}</div>`;
    li.addEventListener("click", () => openVideo(video.video_id));
    list.appendChild(li);
  });
}

async function openVideo(videoId) {
  const record = await apiFetch(`/api/videos/${videoId}`);
  state.currentVideo = record;
  state.videoId = videoId;
  state.scope = "video";
  document.querySelectorAll("#video-list li").forEach((li) => {
    li.classList.toggle("active", li.dataset.id === videoId);
  });
  $("empty-state").classList.add("hidden");
  $("detail").classList.remove("hidden");
  $("detail").classList.remove("scope-global");
  $("delete-video").classList.remove("hidden");
  document.querySelectorAll(".tabs .tab").forEach((tab) => {
    tab.classList.remove("hidden");
  });
  $("video-title").textContent = record.title || videoId;
  const meta = [];
  if (record.author) meta.push(record.author);
  if (record.duration) meta.push(fmtTime(record.duration));
  if (record.transcript_source) meta.push(`via ${record.transcript_source}`);
  if (record.segment_count) meta.push(`${record.segment_count} segments`);
  $("video-meta").textContent = meta.join(" · ");

  const player = $("player");
  if (record.media_url) {
    player.classList.remove("hidden");
    $("no-media").classList.add("hidden");
    if (player.dataset.current !== record.media_url) {
      player.src = record.media_url;
      player.dataset.current = record.media_url;
    }
  } else {
    player.pause();
    player.removeAttribute("src");
    player.dataset.current = "";
    player.classList.add("hidden");
    $("no-media").classList.remove("hidden");
  }

  renderSummary(record.summary);
  renderQuestions(record.summary);
  renderTranscript(record);
  switchTab("summary");
  renderPlaylists();
  await loadThreads();
  loadVideos();
}

async function openGlobal() {
  state.scope = "global";
  state.videoId = null;
  state.currentVideo = null;
  document.querySelectorAll("#video-list li").forEach((li) => {
    li.classList.remove("active");
  });
  $("empty-state").classList.add("hidden");
  $("detail").classList.remove("hidden");
  $("detail").classList.add("scope-global");
  $("delete-video").classList.add("hidden");
  document.querySelectorAll(".tabs .tab").forEach((tab) => {
    tab.classList.toggle("hidden", tab.dataset.tab !== "qa");
  });
  $("video-title").textContent = "All videos";
  $("video-meta").textContent = "conversations across your whole library";
  const player = $("player");
  player.pause();
  player.removeAttribute("src");
  player.dataset.current = "";
  player.classList.add("hidden");
  $("no-media").classList.add("hidden");
  setMode("chat");
  switchTab("qa");
  renderPlaylists();
  await loadThreads();
}

function renderSummary(summary) {
  $("summary-model").textContent = summary ? `(${summary.model || "?"})` : "";
  const target = $("summary-text");
  if (!summary || !summary.summary) {
    target.innerHTML = '<span class="muted">Not summarized yet.</span>';
    return;
  }
  target.innerHTML = linkifyTimestamps(escapeHtml(summary.summary));
}

function renderQuestions(summary) {
  const list = $("questions-list");
  list.innerHTML = "";
  const questions = (summary && summary.highlights) || [];
  if (!questions.length) {
    const li = document.createElement("li");
    li.className = "muted";
    li.textContent = "No questions generated.";
    list.appendChild(li);
    return;
  }
  questions.forEach((question) => {
    const li = document.createElement("li");
    li.textContent = question;
    li.addEventListener("click", () => {
      switchTab("qa");
      $("ask-input").value = question;
      $("ask-input").focus();
    });
    list.appendChild(li);
  });
}

function renderTranscript(record) {
  $("transcript-source").textContent = record.transcript_source
    ? `(via ${record.transcript_source})` : "";
  const target = $("transcript-list");
  target.innerHTML = "";
  (record.chunks || []).forEach((chunk) => {
    const div = document.createElement("div");
    div.className = "chunk";
    const stamp = document.createElement("span");
    stamp.className = "stamp";
    stamp.textContent = chunk.start || chunk.end ? fmtTime(chunk.start) : "—";
    stamp.addEventListener("click", () => seekTo(chunk.start));
    const text = document.createElement("span");
    text.innerHTML = linkifyTimestamps(escapeHtml(chunk.text));
    div.appendChild(stamp);
    div.appendChild(text);
    target.appendChild(div);
  });
  if (!(record.chunks || []).length) {
    target.innerHTML = '<span class="muted">No transcript chunks.</span>';
  }
}

function threadUrl(suffix = "") {
  return state.scope === "global"
    ? `/api/threads${suffix}`
    : `/api/videos/${state.videoId}/threads${suffix}`;
}

function summarizeThread(thread) {
  return {
    thread_id: thread.thread_id,
    title: thread.title,
    mode: thread.mode,
    message_count: (thread.messages || []).length,
  };
}

async function loadThreads(preferredId = null) {
  state.threads = await apiFetch(threadUrl());
  if (!state.threads.length) {
    const created = await apiFetch(threadUrl(), {
      method: "POST",
      body: JSON.stringify({ mode: state.mode }),
    });
    state.threads = [summarizeThread(created)];
  }
  const select = $("thread-select");
  select.innerHTML = "";
  state.threads.forEach((thread) => {
    const option = document.createElement("option");
    option.value = thread.thread_id;
    option.textContent = thread.title || "New conversation";
    select.appendChild(option);
  });
  const wanted = preferredId && state.threads.some((t) => t.thread_id === preferredId)
    ? preferredId
    : state.threads[0].thread_id;
  select.value = wanted;
  await loadThread(wanted);
}

async function loadThread(threadId) {
  const thread = await apiFetch(`/api/threads/${threadId}`);
  state.threadId = threadId;
  state.currentThread = thread;
  setMode(thread.mode || "ask");
  renderScope();
  renderChat(thread.messages || []);
  refreshContext();
}

function setMode(mode) {
  state.mode = mode === "chat" ? "chat" : "ask";
  document.querySelectorAll("#mode-toggle button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === state.mode);
  });
}

function renderChat(messages) {
  const chat = $("chat");
  chat.innerHTML = "";
  if (!messages.length) {
    const hint = document.createElement("div");
    hint.className = "msg muted";
    hint.textContent = state.scope === "global"
      ? "Ask across everything you've downloaded — answers cite the video and moment."
      : "Ask a question, or just talk through what you're learning.";
    chat.appendChild(hint);
    return;
  }
  messages.forEach((message) => {
    if (message.role === "user") {
      appendMessage("user", message.content);
    } else {
      appendBot(message.content, message.citations || []);
    }
  });
}

function appendMessage(kind, text) {
  const chat = $("chat");
  const div = document.createElement("div");
  div.className = `msg ${kind}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function citationLabel(citation) {
  const stamp = fmtTime(citation.start);
  if (citation.video_title && citation.video_id && citation.video_id !== state.videoId) {
    const title = citation.video_title.length > 26
      ? citation.video_title.slice(0, 26) + "…"
      : citation.video_title;
    return `${title} · ${stamp}`;
  }
  return `${stamp} (${Math.round((citation.score || 0) * 100)}%)`;
}

function openCitation(citation) {
  if (citation.video_id && citation.video_id !== state.videoId) {
    openVideo(citation.video_id).then(() => seekTo(citation.start));
  } else {
    seekTo(citation.start);
  }
}

function appendBot(text, citations) {
  const chat = $("chat");
  const div = document.createElement("div");
  div.className = "msg bot";
  div.innerHTML = linkifyTimestamps(escapeHtml(text));
  if (citations && citations.length) {
    const row = document.createElement("div");
    row.className = "citations";
    citations.forEach((citation) => {
      const chip = document.createElement("button");
      chip.className = "citation-chip";
      chip.textContent = citationLabel(citation);
      chip.title = citation.text;
      chip.addEventListener("click", () => openCitation(citation));
      row.appendChild(chip);
    });
    div.appendChild(row);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

// ---------- tabs ------------------------------------------------------------

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${name}`);
  });
}

// ---------- events ----------------------------------------------------------

function bindEvents() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  document.addEventListener("click", (event) => {
    const link = event.target.closest(".ts-link");
    if (link) seekTo(Number(link.dataset.seconds || 0));
  });

  $("refresh-models").addEventListener("click", refreshModels);
  $("pull-model").addEventListener("click", pullModel);
  $("model-select").addEventListener("change", (event) => switchModel(event.target.value));
  $("open-global").addEventListener("click", openGlobal);

  $("thread-select").addEventListener("change", (event) => loadThread(event.target.value));

  document.querySelectorAll("#mode-toggle button").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  $("new-playlist").addEventListener("click", createPlaylist);

  $("scope-toggle").addEventListener("click", (event) => {
    event.stopPropagation();
    $("scope-menu").classList.toggle("hidden");
  });

  $("scope-all").addEventListener("change", (event) => {
    const disabled = event.target.checked;
    $("scope-playlists").querySelectorAll("input").forEach((box) => {
      box.disabled = disabled;
      if (disabled) box.checked = false;
    });
  });

  $("scope-apply").addEventListener("click", applyScope);

  document.addEventListener("click", (event) => {
    if ($("scope-menu") && !event.target.closest("#scope-picker")) {
      $("scope-menu").classList.add("hidden");
    }
  });

  $("context-btn").addEventListener("click", () => {
    $("context-modal").classList.remove("hidden");
    refreshContext();
  });

  $("context-close").addEventListener("click", () => {
    $("context-modal").classList.add("hidden");
  });

  $("context-modal").addEventListener("click", (event) => {
    if (event.target === $("context-modal")) {
      $("context-modal").classList.add("hidden");
    }
  });

  $("new-thread").addEventListener("click", async () => {
    const created = await apiFetch(threadUrl(), {
      method: "POST",
      body: JSON.stringify({ mode: state.mode }),
    });
    await loadThreads(created.thread_id);
  });

  $("delete-thread").addEventListener("click", async () => {
    if (!state.threadId) return;
    if (!confirm("Delete this conversation?")) return;
    await apiFetch(`/api/threads/${state.threadId}`, { method: "DELETE" });
    state.threadId = null;
    await loadThreads();
  });

  $("add-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const url = $("url-input").value.trim();
    if (!url) return;
    requestNotifyPermission();
    $("add-btn").disabled = true;
    showJob("starting download…");
    try {
      const { job_id } = await apiFetch("/api/videos", {
        method: "POST",
        body: JSON.stringify({
          url,
          video: $("opt-video").checked,
          audio: $("opt-audio").checked,
          subtitles: $("opt-subs").checked,
          force: $("opt-force").checked,
          force_subtitles: $("opt-subtitles-only").checked,
        }),
      });
      pollJob(job_id, {
        onDone: async (job) => {
          notifyFinished(job);
          $("add-btn").disabled = false;
          $("url-input").value = "";
          await loadVideos();
          if (job.result && job.result.video_id) openVideo(job.result.video_id);
        },
        onError: () => { $("add-btn").disabled = false; },
      });
    } catch (error) {
      $("add-btn").disabled = false;
      showJobError(error.message);
    }
  });

  $("reload-videos").addEventListener("click", loadVideos);

  $("delete-video").addEventListener("click", async () => {
    if (!state.videoId) return;
    if (!confirm(`Delete "${state.currentVideo?.title || state.videoId}" and its files?`)) return;
    await apiFetch(`/api/videos/${state.videoId}`, { method: "DELETE" });
    $("detail").classList.add("hidden");
    $("empty-state").classList.remove("hidden");
    state.videoId = null;
    state.currentVideo = null;
    await loadVideos();
    await loadPlaylists();
  });

  $("ask-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = $("ask-input").value.trim();
    if (!question || !state.threadId) return;
    $("ask-input").value = "";
    $("ask-btn").disabled = true;
    appendMessage("user", question);
    const pending = appendMessage("bot pending", "thinking…");
    try {
      const payload = await apiFetch(`/api/threads/${state.threadId}/messages`, {
        method: "POST",
        body: JSON.stringify({ message: question, mode: state.mode }),
      });
      pending.remove();
      const message = payload.message || {};
      appendBot(message.content || "", message.citations || []);
      if (payload.context) renderContext(payload.context);
      const thread = payload.thread || {};
      if (thread.title) {
        const option = $("thread-select").selectedOptions[0];
        if (option) option.textContent = thread.title;
      }
    } catch (error) {
      pending.remove();
      appendBot(`Error: ${error.message}`, []);
    } finally {
      $("ask-btn").disabled = false;
    }
  });

  const player = $("player");
  player.addEventListener("timeupdate", () => {
    $("time-chip").textContent = fmtTime(player.currentTime);
  });
}

// ---------- init ------------------------------------------------------------

async function init() {
  bindEvents();
  await refreshModels();
  await loadVideos();
  await loadPlaylists();
  setInterval(refreshModels, 30000);
}

document.addEventListener("DOMContentLoaded", init);
