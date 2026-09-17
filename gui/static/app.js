"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  currentVideo: null,
  videoId: null,
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
      onDone: async () => { await refreshModels(); },
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
  document.querySelectorAll("#video-list li").forEach((li) => {
    li.classList.toggle("active", li.dataset.id === videoId);
  });
  $("empty-state").classList.add("hidden");
  $("detail").classList.remove("hidden");
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
  renderChat(record.answers || []);
  switchTab("summary");
  loadVideos();
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

function renderChat(answers) {
  const chat = $("chat");
  chat.innerHTML = "";
  answers.forEach((entry) => {
    appendMessage("user", entry.question);
    appendBot(entry.answer, entry.citations || []);
  });
  chat.scrollTop = chat.scrollHeight;
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
      chip.textContent = `${fmtTime(citation.start)} (${Math.round((citation.score || 0) * 100)}%)`;
      chip.title = citation.text;
      chip.addEventListener("click", () => seekTo(citation.start));
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

  $("add-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const url = $("url-input").value.trim();
    if (!url) return;
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
  });

  $("ask-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = $("ask-input").value.trim();
    if (!question || !state.videoId) return;
    $("ask-input").value = "";
    $("ask-btn").disabled = true;
    appendMessage("user", question);
    const pending = appendMessage("bot pending", "searching the transcript…");
    try {
      const payload = await apiFetch(`/api/videos/${state.videoId}/ask`, {
        method: "POST",
        body: JSON.stringify({ question }),
      });
      pending.remove();
      const answer = payload.answer || {};
      appendBot(answer.answer || "", answer.citations || []);
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
  setInterval(refreshModels, 30000);
}

document.addEventListener("DOMContentLoaded", init);
