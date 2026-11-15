/**
 * Aufgabe 9 – Texturierter Torus
 *
 * Der Torus erhält zu Position und Normale Texturkoordinaten (u, v) ∈ [0, 1]²:
 *   u läuft einmal um die Torusachse, v einmal um die Röhre.
 * Drei Darstellungen im Fragment-Shader:
 *   1  Bildtextur  – selbst erzeugter Marmor (textures/marmor.jpg), per texture2D abgetastet
 *   2  Prozedural – Azulejo-Kacheln, im Shader aus u und v berechnet (Erweiterung)
 *   3  UV-Koordinaten als Farbe (zur Kontrolle)
 * Taste F zeigt den Torus aufgeschnitten als Ebene (Abwicklung) mit denselben Texturkoordinaten.
 */
(() => {
  "use strict";

  // Vektoren und Matrizen
  const PI = Math.PI, TAU = 2 * PI;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

  const M4 = {
    perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
      return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
    },
    lookAt(eye, center, up) {
      const z = norm(sub(eye, center)), x = norm(cross(up, z)), y = cross(z, x);
      return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
    },
  };

  // Geometrie mit Texturkoordinaten
  const TORUS = { R: 1, r: 0.42 };

  /**
   * Torus (liegt in der XZ-Ebene). Je Vertex: Position, Normale und Texturkoordinate (u, v).
   * Die erste und letzte Spalte/Zeile liegen auf derselben Stelle, haben aber u = 0 bzw. u = 1 –
   * so läuft die Textur ohne Sprung einmal ganz herum.
   */
  function torus(nU = 120, nV = 48) {
    const { R, r } = TORUS, pos = [], nor = [], uv = [], idx = [];
    for (let i = 0; i <= nU; i++) for (let j = 0; j <= nV; j++) {
      const u = i / nU, v = j / nV, a = u * TAU, b = v * TAU;
      const n = [Math.cos(b) * Math.cos(a), Math.sin(b), Math.cos(b) * Math.sin(a)];
      pos.push((R + r * Math.cos(b)) * Math.cos(a), r * Math.sin(b), (R + r * Math.cos(b)) * Math.sin(a));
      nor.push(...n);
      uv.push(u, v);                                  // ← die neuen Texturkoordinaten
    }
    for (let i = 0; i < nU; i++) for (let j = 0; j < nV; j++) {
      const a = i * (nV + 1) + j, c = a + nV + 1;
      idx.push(a, a + 1, c, a + 1, c + 1, c);
    }
    return { pos, nor, uv, idx };
  }

  /** Abwicklung: der Torus aufgeschnitten als Rechteck (Breite 2πR, Höhe 2πr), gleiche (u, v) */
  function unrolled(nU = 60, nV = 24) {
    const { R, r } = TORUS, Wd = TAU * R, Hd = TAU * r, pos = [], nor = [], uv = [], idx = [];
    for (let i = 0; i <= nU; i++) for (let j = 0; j <= nV; j++) {
      const u = i / nU, v = j / nV;
      pos.push((u - 0.5) * Wd, 0.25 + v * Hd, 0); nor.push(0, 0, 1); uv.push(u, v);
    }
    for (let i = 0; i < nU; i++) for (let j = 0; j < nV; j++) {
      const a = i * (nV + 1) + j, c = a + nV + 1;
      idx.push(a, c, a + 1, a + 1, c, c + 1);
    }
    return { pos, nor, uv, idx };
  }

  // Shader
  const VS = `
    attribute vec3 aPos;
    attribute vec3 aNormal;
    attribute vec2 aUV;
    uniform mat4 uView, uProj;
    varying vec3 vPos;
    varying vec3 vNormal;
    varying vec2 vUV;
    void main() {
      vPos = aPos;                                   // Modellmatrix = Einheitsmatrix
      vNormal = aNormal;
      vUV = aUV;                                     // Texturkoordinate wird interpoliert
      gl_Position = uProj * uView * vec4(aPos, 1.0);
    }`;
  const FS = `
    precision highp float;
    varying vec3 vPos;
    varying vec3 vNormal;
    varying vec2 vUV;
    uniform sampler2D uTex;
    uniform int uMode;                               // 0 Bild, 1 prozedural, 2 UV
    uniform vec2 uRepeat;                            // Wiederholungen der Bildtextur
    uniform vec2 uTiles;                             // Kachelanzahl der prozeduralen Textur
    uniform vec3 uEye;
    uniform vec3 uLight;

    const float PI = 3.14159265;

    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
    // weiche Linie der Breite w um den Wert 0
    float line(float d, float w) { return 1.0 - smoothstep(w * 0.6, w, abs(d)); }

    // ---------------------------------------------------- prozedurale Textur: Azulejo-Kacheln
    vec3 azulejo(vec2 uv, out float glaze) {
      vec2 g = uv * uTiles;                          // ganze Zahlen von Kacheln → nahtlos
      vec2 cell = floor(g);
      vec2 p = fract(g) * 2.0 - 1.0;                 // Kachel-Koordinaten in [-1, 1]
      cell = mod(cell, uTiles);                      // gleiche Kachel bei u = 0 und u = 1
      float h = hash(cell);
      // jede zweite Kachel gespiegelt → größeres Muster über 2 × 2 Kacheln
      if (mod(cell.x, 2.0) > 0.5) p.x = -p.x;
      if (mod(cell.y, 2.0) > 0.5) p.y = -p.y;

      vec3 white = mix(vec3(0.95, 0.94, 0.89), vec3(0.88, 0.9, 0.93), h);
      vec3 cobalt = vec3(0.07, 0.2, 0.55) * (0.85 + 0.3 * hash(cell + 7.0));
      vec3 ochre = vec3(0.86, 0.6, 0.16);
      vec3 col = white;

      float r = length(p), a = atan(p.y, p.x);
      // Blüte in der Mitte: vierblättrig, Kontur in Kobaltblau, gefüllt in Ocker
      float petal = r - (0.48 + 0.2 * cos(4.0 * a));
      if (petal < 0.0) col = mix(col, ochre, 0.85);
      col = mix(col, cobalt, line(petal, 0.07));
      col = mix(col, cobalt, 1.0 - smoothstep(0.12, 0.16, r));     // Mittelpunkt
      // Viertelkreise in den Ecken: bilden mit den Nachbarkacheln einen Ring
      vec2 q = abs(p) - 1.0;
      float rc = length(q);
      col = mix(col, cobalt, line(rc - 0.55, 0.08));
      if (rc < 0.32) col = mix(col, cobalt, 0.9);
      // feine Rahmenlinie
      float frame = max(abs(p.x), abs(p.y));
      col = mix(col, cobalt, line(frame - 0.9, 0.025) * 0.7);
      // Fugen zwischen den Kacheln
      float joint = smoothstep(0.955, 0.985, frame);
      col = mix(col, vec3(0.62, 0.6, 0.56), joint);
      glaze = 1.0 - joint;                           // Fugen glänzen nicht
      // leichte Wölbung der Glasur zum Rand hin
      col *= 1.0 - 0.12 * smoothstep(0.7, 0.95, frame) * glaze;
      return col;
    }

    void main() {
      vec3 N = normalize(vNormal);
      vec3 V = normalize(uEye - vPos);
      if (dot(N, V) < 0.0) N = -N;
      vec3 L = normalize(uLight - vPos);
      float diff = max(dot(N, L), 0.0);
      float spec = pow(max(dot(reflect(-L, N), V), 0.0), 40.0);

      vec3 base; float gloss = 0.35;
      if (uMode == 0) {
        base = texture2D(uTex, vUV * uRepeat).rgb;   // Bildtextur
        gloss = 0.5;                                 // polierter Marmor
      } else if (uMode == 1) {
        float glaze;
        base = azulejo(vUV, glaze);
        gloss = 0.65 * glaze;
      } else {
        base = vec3(vUV, 0.25);                      // u → Rot, v → Grün
        vec2 gl = abs(fract(vUV * 8.0 + 0.5) - 0.5);
        base = mix(base, vec3(1.0), 1.0 - smoothstep(0.0, 0.04, min(gl.x, gl.y)));
        gloss = 0.0;
      }
      vec3 color = base * (0.32 + 0.78 * diff) + vec3(gloss * spec);
      gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
    }`;

  // WebGL
  const $ = (id) => document.getElementById(id);
  const canvas = $("gl");
  const status = $("status");
  const setStatus = (t, err = false) => { status.textContent = t; status.classList.toggle("is-error", err); };
  let gl, loc, meshTorus, meshFlat, texture, textureReady = false;

  function compile(type, src) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(`Shader-Fehler: ${gl.getShaderInfoLog(sh)}`);
    return sh;
  }
  function upload(m) {
    const mk = (t, d) => { const b = gl.createBuffer(); gl.bindBuffer(t, b); gl.bufferData(t, d, gl.STATIC_DRAW); return b; };
    if (m.pos.length / 3 > 65535) throw new Error("Zu viele Vertices für UNSIGNED_SHORT");
    return {
      pos: mk(gl.ARRAY_BUFFER, new Float32Array(m.pos)), nor: mk(gl.ARRAY_BUFFER, new Float32Array(m.nor)),
      uv: mk(gl.ARRAY_BUFFER, new Float32Array(m.uv)), idx: mk(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(m.idx)),
      n: m.idx.length, verts: m.pos.length / 3,
    };
  }

  /** Bildtextur laden. Bis das Bild da ist, steht ein einfarbiges Platzhalter-Texel bereit. */
  function loadTexture(url) {
    texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([40, 110, 120, 255]));
    const img = new Image();
    img.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      // 1024 × 512 ist eine Zweierpotenz → Wiederholung und Mipmaps sind in WebGL 1 erlaubt
      gl.generateMipmap(gl.TEXTURE_2D);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      const aniso = gl.getExtension("EXT_texture_filter_anisotropic");
      if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, Math.min(8, gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
      textureReady = true;
      $("ro-tex").textContent = `${img.naturalWidth} × ${img.naturalHeight} px, geladen`;
      redraw();
    };
    img.onerror = () => {
      $("ro-tex").textContent = "nicht geladen";
      setStatus(`Die Bildtextur „${url}“ konnte nicht geladen werden. Wird die Seite direkt als Datei geöffnet, bitte einen lokalen Webserver starten (z. B. python3 -m http.server).`, true);
      console.error(`Textur nicht geladen: ${url}`);
    };
    img.src = url;
  }

  function init() {
    gl = canvas.getContext("webgl", { antialias: true });
    if (!gl) throw new Error("Ihr Browser unterstützt kein WebGL. Bitte aktivieren Sie die Hardwarebeschleunigung.");
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`Programm-Fehler: ${gl.getProgramInfoLog(prog)}`);
    gl.useProgram(prog);
    loc = {};
    ["aPos", "aNormal", "aUV"].forEach((n) => (loc[n] = gl.getAttribLocation(prog, n)));
    ["uView", "uProj", "uTex", "uMode", "uRepeat", "uTiles", "uEye", "uLight"].forEach((n) => (loc[n] = gl.getUniformLocation(prog, n)));
    gl.enable(gl.DEPTH_TEST);
    meshTorus = upload(torus());
    meshFlat = upload(unrolled());
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(loc.uTex, 0);
    loadTexture("textures/marmor.jpg");
  }

  // Zustand
  const CAM0 = { angle: 20, radius: 4.2, height: 2.0 };
  const CAM_FLAT = { angle: 0, radius: 7.2, height: 1.8 };
  const cam = { ...CAM0 };
  const LIMITS = { radius: [2.2, 9], height: [-3, 5] };
  const MODES = ["bild", "prozedural", "uv"];
  const MODE_LABEL = { bild: "Bildtextur (Marmor)", prozedural: "Prozedurale Textur (Azulejo-Kacheln)", uv: "Texturkoordinaten als Farbe" };
  const state = { mode: "bild", flat: false, repeat: 1, tilesU: 36, tilesV: 12 };

  const eyePos = () => { const a = (cam.angle * PI) / 180; return [cam.radius * Math.sin(a), cam.height, cam.radius * Math.cos(a)]; };

  // Zeichnen
  function draw() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    gl.clearColor(...(dark ? [0.11, 0.13, 0.17] : [0.93, 0.94, 0.95]), 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const eye = eyePos(), target = state.flat ? [0, 1.55, 0] : [0, 0, 0];
    gl.uniformMatrix4fv(loc.uProj, false, new Float32Array(M4.perspective((45 * PI) / 180, w / h, 0.1, 50)));
    gl.uniformMatrix4fv(loc.uView, false, new Float32Array(M4.lookAt(eye, target, [0, 1, 0])));
    gl.uniform3fv(loc.uEye, eye);
    gl.uniform3fv(loc.uLight, [3, 5, 4]);           // Licht fest in der Welt
    gl.uniform1i(loc.uMode, MODES.indexOf(state.mode));
    gl.uniform2fv(loc.uRepeat, [state.repeat, 1]);
    gl.uniform2fv(loc.uTiles, [state.tilesU, state.tilesV]);
    gl.bindTexture(gl.TEXTURE_2D, texture);

    const m = state.flat ? meshFlat : meshTorus;
    [["aPos", m.pos, 3], ["aNormal", m.nor, 3], ["aUV", m.uv, 2]].forEach(([n, b, size]) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.enableVertexAttribArray(loc[n]); gl.vertexAttribPointer(loc[n], size, gl.FLOAT, false, 0, 0);
    });
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.idx);
    gl.drawElements(gl.TRIANGLES, m.n, gl.UNSIGNED_SHORT, 0);

    const err = gl.getError();
    if (err !== gl.NO_ERROR) console.error(`WebGL-Fehler 0x${err.toString(16)}`);
    const f = (v) => v.toLocaleString("de-DE", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    $("ro-cam").textContent = `(${eye.map(f).join(" | ")})`;
  }

  function renderState() {
    document.querySelectorAll("[data-mode]").forEach((b) => { const on = b.dataset.mode === state.mode; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; });
    $("btn-flat").setAttribute("aria-pressed", String(state.flat));
    $("repeat").value = state.repeat; $("repeat-out").textContent = `${state.repeat}×`;
    $("tiles-u").value = state.tilesU; $("tiles-u-out").textContent = String(state.tilesU);
    $("tiles-v").value = state.tilesV; $("tiles-v-out").textContent = String(state.tilesV);
    $("row-repeat").hidden = state.mode !== "bild";
    $("row-tiles").hidden = state.mode !== "prozedural";
    $("ro-mode").textContent = MODE_LABEL[state.mode];
    $("ro-form").textContent = state.flat ? "Abwicklung (Ebene)" : "Torus";
    $("canvas-desc").textContent =
      `${state.flat ? "Der aufgeschnittene Torus als flaches Rechteck" : "Ein Torus"} mit ${MODE_LABEL[state.mode]}. ` +
      ({ bild: "Türkis-dunkelblauer Marmor mit weißen und goldenen Adern.", prozedural: `Weiße Kacheln mit kobaltblauen Blüten, ockerfarbener Füllung und Ringen über den Kachelecken, ${state.tilesU} × ${state.tilesV} Kacheln.`, uv: "u als Rotanteil, v als Grünanteil, weiße Gitterlinien in Achtelschritten." })[state.mode];
  }

  // Aktionen
  let frame = null;
  const redraw = () => { if (frame) return; frame = requestAnimationFrame(() => { frame = null; draw(); }); };
  function setMode(m, announce = true) {
    state.mode = m; renderState(); redraw();
    if (announce) setStatus(`${MODE_LABEL[m]}${m === "bild" && !textureReady ? " – Bild wird noch geladen" : ""}.`);
  }
  function setFlat(on) { state.flat = on; Object.assign(cam, on ? CAM_FLAT : CAM0); renderState(); redraw(); setStatus(on ? "Abwicklung: der Torus aufgeschnitten als Ebene." : "Wieder als Torus."); }
  function orbit(d) { cam.angle += d; redraw(); }
  function changeHeight(d) { cam.height = clamp(cam.height + d, ...LIMITS.height); redraw(); }
  function changeRadius(d) { cam.radius = clamp(cam.radius + d, ...LIMITS.radius); redraw(); }
  function reset() { Object.assign(cam, state.flat ? CAM_FLAT : CAM0); redraw(); setStatus("Kamera zurückgesetzt."); }

  // Bedienung
  document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("mode-group").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault(); e.stopPropagation();
    const d = e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1;
    const next = MODES[(MODES.indexOf(state.mode) + d + MODES.length) % MODES.length];
    setMode(next); document.querySelector(`[data-mode="${next}"]`).focus();
  });
  $("btn-flat").addEventListener("click", () => setFlat(!state.flat));
  const slider = (id, key, unit = "") => {
    $(id).addEventListener("input", (e) => { state[key] = Number(e.target.value); renderState(); redraw(); });
    $(id).addEventListener("change", () => setStatus(`${$(id).labels[0].textContent}: ${state[key]}${unit}.`));
  };
  slider("repeat", "repeat", "×"); slider("tiles-u", "tilesU"); slider("tiles-v", "tilesV");
  $("btn-left").addEventListener("click", () => orbit(-10));
  $("btn-right").addEventListener("click", () => orbit(10));
  $("btn-up").addEventListener("click", () => changeHeight(0.3));
  $("btn-down").addEventListener("click", () => changeHeight(-0.3));
  $("btn-near").addEventListener("click", () => changeRadius(-0.4));
  $("btn-far").addEventListener("click", () => changeRadius(0.4));
  $("btn-reset").addEventListener("click", reset);

  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    orbit(-(e.clientX - drag.x) * 0.4); changeHeight((e.clientY - drag.y) * 0.012);
    drag = { x: e.clientX, y: e.clientY };
  });
  ["pointerup", "pointercancel"].forEach((t) => canvas.addEventListener(t, () => (drag = null)));
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); changeRadius(Math.sign(e.deltaY) * 0.3); }, { passive: false });

  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* privater Modus */ } },
  };
  const shortcuts = $("toggle-shortcuts");
  shortcuts.checked = store.get("shortcuts") !== "off";
  shortcuts.addEventListener("change", () => {
    store.set("shortcuts", shortcuts.checked ? "on" : "off");
    setStatus(shortcuts.checked ? "Tastenkürzel eingeschaltet." : "Tastenkürzel ausgeschaltet – Bedienung über die Schaltflächen.");
  });

  document.addEventListener("keydown", (e) => {
    if (!shortcuts.checked || e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target;
    if (t.closest && t.closest("input, textarea, select, [role=radiogroup]")) return;
    const k = e.key;
    if (k.startsWith("Arrow") && t.closest && (t.closest("#buehne") || t === document.body)) {
      e.preventDefault();
      if (k === "ArrowLeft") orbit(-5);
      if (k === "ArrowRight") orbit(5);
      if (k === "ArrowUp") changeHeight(0.2);
      if (k === "ArrowDown") changeHeight(-0.2);
      return;
    }
    switch (k.toLowerCase()) {
      case "a": e.preventDefault(); orbit(-5); break;
      case "d": e.preventDefault(); orbit(5); break;
      case "w": e.preventDefault(); changeHeight(0.2); break;
      case "s": e.preventDefault(); changeHeight(-0.2); break;
      case "n": e.preventDefault(); changeRadius(e.shiftKey ? -0.3 : 0.3); break;
      case "r": if (!e.repeat) { e.preventDefault(); reset(); } break;
      case "m": if (!e.repeat) { e.preventDefault(); setMode(MODES[(MODES.indexOf(state.mode) + 1) % MODES.length]); } break;
      case "1": case "2": case "3": if (!e.repeat) { e.preventDefault(); setMode(MODES[Number(k) - 1]); } break;
      case "f": if (!e.repeat) { e.preventDefault(); setFlat(!state.flat); } break;
    }
  });

  const themeSwitch = $("theme-switch");
  function applyTheme(th) {
    document.documentElement.setAttribute("data-theme", th);
    themeSwitch.setAttribute("aria-checked", String(th === "dark"));
    if (gl) redraw();
  }
  themeSwitch.addEventListener("click", () => {
    const next = themeSwitch.getAttribute("aria-checked") === "true" ? "light" : "dark";
    applyTheme(next); store.set("theme", next);
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => { if (!store.get("theme")) applyTheme(e.matches ? "dark" : "light"); });

  // Start
  applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light");
  try {
    init();
    $("info-verts").textContent = meshTorus.verts.toLocaleString("de-DE");
    $("info-tris").textContent = (meshTorus.n / 3).toLocaleString("de-DE");
    renderState();
    draw();
    setStatus("");
    new ResizeObserver(() => redraw()).observe(canvas);
    canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); setStatus("WebGL-Kontext verloren – bitte Seite neu laden.", true); });
    console.info(`Torus: ${meshTorus.verts} Vertices, ${meshTorus.n / 3} Dreiecke.`);
  } catch (err) {
    console.error(err);
    setStatus(err.message, true);
  }
})();
