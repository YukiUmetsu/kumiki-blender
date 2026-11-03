"""Dependency-free, standalone browser annotation interface.

No CDN, uploads, analytics, camera access, or server required. Generated
documents contain *private photo previews* and must stay out of the Git repo.
"""

TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; object-src 'none'; form-action 'none'">
<title>Kumiki | Statue face annotation</title>
<style>
  :root { --bg:#101820; --surface:#1a252f; --surface-2:#22313d; --border:#344653;
    --text:#f0f5f7; --muted:#aec1cc; --teal:#67e4d5; --orange:#efbd72;
    --accent:#6de1ce; --focus:#e5ca76; --radius:12px; font-family:
    -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  * { box-sizing:border-box; }
  [hidden] { display:none !important; }
  body { margin:0; background:var(--bg); color:var(--text); font-size:14px; line-height:1.4; }
  button,select,input { font:inherit; }
  button,select { min-height:40px; border-radius:9px; border:1px solid var(--border);
    background:var(--surface-2); color:var(--text); }
  button { cursor:pointer; padding:8px 12px; }
  button:hover:not(:disabled) { border-color:#718c9a; background:#2a3e4c; }
  button:active:not(:disabled) { transform:translateY(1px); }
  button:disabled { opacity:.5; cursor:not-allowed; }
  :focus-visible { outline:3px solid var(--focus); outline-offset:2px; }
  .topbar { height:66px; display:flex; align-items:center; gap:12px; padding:0 20px;
    border-bottom:1px solid var(--border); background:var(--surface); }
  .brand { min-width:0; margin-right:auto; display:flex; gap:10px; align-items:center; }
  .brandmark { display:grid; place-items:center; background:#24473f; color:var(--teal);
    width:34px; height:34px; border-radius:10px; font-weight:800; }
  h1 { font-size:16px; margin:0; font-weight:650; line-height:1.25; }
  .subtle { font-size:12px; color:var(--muted); }
  .top-actions { display:flex; align-items:center; gap:8px; }
  .primary { background:var(--accent); color:#092b28; font-weight:700; border-color:transparent; }
  .primary:hover:not(:disabled) { background:#9af5e8; border-color:transparent; }
  .workspace { display:grid; grid-template-columns:172px minmax(360px,1fr) 268px;
    height:calc(100dvh - 66px); min-height:480px; }
  .filmstrip,.tools { background:var(--surface); overflow:auto; }
  .filmstrip { padding:14px 10px; border-right:1px solid var(--border); }
  .tools { padding:18px 16px; border-left:1px solid var(--border); }
  .section-title { font-size:11px; letter-spacing:.09em; text-transform:uppercase;
    color:var(--muted); font-weight:700; margin:2px 6px 12px; }
  .photo-item { width:100%; padding:6px; text-align:left; margin:0 0 8px; display:block; background:transparent;
    border:1px solid transparent; border-radius:10px; }
  .photo-item img { width:100%; height:100px; object-fit:contain; border-radius:6px;
    background:#111c26; display:block; }
  .photo-item[aria-current="true"] { background:#1b393a; border-color:var(--teal); }
  .photo-name { display:block; margin-top:6px; font-weight:600; font-size:11px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .photo-view { display:block; color:var(--muted); font-size:11px; margin-top:2px; }
  .center { display:flex; min-width:0; min-height:0; flex-direction:column; }
  .toolbar { min-height:68px; display:flex; align-items:center; flex-wrap:wrap; gap:8px;
    padding:10px 14px; border-bottom:1px solid var(--border); background:#16232d; }
  .field { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:12px; }
  .field select { min-width:154px; padding:7px 10px; }
  .field input[type="range"] { width:92px; accent-color:var(--teal); }
  .separator { width:1px; height:28px; background:var(--border); margin:0 3px; }
  .is-active { color:var(--teal); background:#20413e; border-color:var(--teal); }
  .viewer { overflow:auto; min-height:240px; flex:1; background:#0a1118; position:relative;
    background-image:linear-gradient(45deg,#0e1a24 25%,transparent 25%),
      linear-gradient(-45deg,#0e1a24 25%,transparent 25%),
      linear-gradient(45deg,transparent 75%,#0e1a24 75%),
      linear-gradient(-45deg,transparent 75%,#0e1a24 75%);
    background-size:28px 28px; background-position:0 0,0 14px,14px -14px,-14px 0;
  }
  .image-stage { position:relative; margin:20px auto; cursor:crosshair; user-select:none; touch-action:manipulation; }
  .image-stage img { display:block; width:100%; height:100%; pointer-events:none; }
  .image-stage svg { position:absolute; inset:0; width:100%; height:100%; pointer-events:none;
    overflow:visible; }
  .viewer-hint { min-height:42px; padding:12px 16px; border-top:1px solid var(--border);
    color:var(--muted); background:#16232d; font-size:12px; display:flex; gap:12px;
    align-items:center; }
  .viewer-hint [role="status"] { flex:1; }
  .viewer-hint .error { color:#ffc9a8; }
  .tools h2 { margin:0 0 8px; font-size:15px; }
  .tools .muted { font-size:12px; color:var(--muted); margin:0 0 14px; }
  .search { width:100%; min-height:40px; padding:8px 10px; border-radius:9px;
    border:1px solid var(--border); background:#101c25; color:var(--text); margin-bottom:10px; }
  .landmark-list { display:grid; gap:5px; }
  .landmark-btn { text-align:left; border:1px solid transparent; background:transparent;
    font-size:12px; display:flex; justify-content:space-between; gap:10px; align-items:center;
    min-height:37px; }
  .landmark-btn[aria-pressed="true"] { border-color:var(--teal); background:#20413e; }
  .landmark-btn .set { color:var(--teal); font-weight:700; }
  .landmark-btn .suggested { color:var(--orange); }
  .tool-footer { border-top:1px solid var(--border); margin-top:12px; padding-top:12px; }
  .tool-footer button { width:100%; margin-bottom:8px; }
  .visibility-field { margin:12px 0 8px; justify-content:space-between; }
  .visibility-field select { min-width:138px; }
  .kbd { font:11px ui-monospace,SFMono-Regular,Consolas,monospace; color:#c4d5de;
    padding:2px 4px; background:#2b3d48; border-radius:4px; }
  .help { line-height:1.75; font-size:11px; color:var(--muted); margin-top:12px; }
  @media (max-width:1050px) { .workspace { grid-template-columns:124px minmax(320px,1fr) 225px; }
    .photo-item img { height:76px; } .tools { padding:14px 10px; } .topbar { padding:0 12px; } }
  @media (max-width:780px) { .topbar { height:auto; min-height:60px; flex-wrap:wrap; padding:10px; }
    .workspace { height:auto; min-height:0; display:flex; flex-direction:column; }
    .filmstrip { order:0; height:128px; flex:none; border-right:0; border-bottom:1px solid var(--border);
      overflow-x:auto; overflow-y:hidden; padding:8px; }
    .section-title { display:none; } #photos { display:flex; gap:8px; height:108px; }
    .photo-item { width:112px; flex:none; margin:0; } .photo-item img { height:62px; }
    .center { order:1; height:65dvh; min-height:420px; } .tools { order:2; border-left:0;
      border-top:1px solid var(--border); max-height:none; }
    .landmark-list { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .viewer-hint .shortcuts { display:none; } }
  @media (prefers-reduced-motion:reduce) { button:active:not(:disabled) { transform:none; } }
</style>
</head>
<body>
<header class="topbar">
  <div class="brand"><div class="brandmark" aria-hidden="true">K</div>
    <div><h1>Statue face annotation</h1><div class="subtle" id="progress">Local photo set</div></div>
  </div>
  <div class="top-actions">
    <button id="undo" type="button" title="Undo the last annotation edit">Undo</button>
    <button id="export" type="button" class="primary">Export JSON</button>
  </div>
</header>
<main class="workspace">
  <nav class="filmstrip" aria-label="Reference photos">
    <div class="section-title">Reference views</div><div id="photos"></div>
  </nav>
  <section class="center" aria-label="Annotation canvas">
    <div class="toolbar">
      <label class="field">View <select id="view">
        <option value="unknown">Unknown</option><option value="front">Front</option>
        <option value="front-three-quarter">Front ¾</option><option value="profile">Profile</option>
        <option value="rear-three-quarter">Rear ¾</option><option value="rear">Rear</option>
      </select></label>
      <span class="separator" aria-hidden="true"></span>
      <button id="outline" type="button" aria-pressed="false" title="Draw the visible head outline">Outline</button>
      <button id="finish-outline" type="button" disabled>Finish outline</button>
      <label class="field">Zoom <input id="zoom" type="range" min="1" max="3" step="0.25" value="1"
        aria-label="Image zoom"><span id="zoom-value">100%</span></label>
    </div>
    <div id="viewer" class="viewer" tabindex="0" aria-label="Photo with landmark overlay">
      <div id="stage" class="image-stage"><img id="image" alt="Reference statue face">
        <svg id="overlay" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"></svg></div>
    </div>
    <div class="viewer-hint"><span id="status" role="status" aria-live="polite">Choose a landmark, then click on the photo.</span>
      <span class="shortcuts">← → views · <span class="kbd">O</span> outline · <span class="kbd">⌘/Ctrl S</span> export</span></div>
  </section>
  <aside class="tools" aria-label="Landmark selection">
    <h2>Facial points</h2>
    <p class="muted">Select a feature and click its position. A new point replaces the previous one.</p>
    <input id="search" class="search" type="search" placeholder="Find a point…" aria-label="Find a landmark">
    <div id="point-count" class="subtle" aria-live="polite"></div>
    <label class="field visibility-field">Selected point status <select id="point-visibility">
      <option value="unmarked">Not reviewed</option>
      <option value="visible">Visible</option>
      <option value="uncertain">Uncertain</option>
      <option value="occluded">Occluded</option>
      <option value="not-visible">Not visible</option>
    </select></label>
    <div id="landmarks" class="landmark-list"></div>
    <div class="tool-footer">
      <button id="delete-point" type="button">Remove selected point</button>
      <button id="clear-outline" type="button">Clear current outline</button>
      <button id="import" type="button">Import annotation JSON</button>
      <input id="import-file" type="file" accept="application/json,.json" hidden>
      <p class="help">For uncertain or occluded features, place a point first and change its status. Mark absent features “Not visible” without coordinates. Outline: click around the head, then Finish or Enter. Undo works for all edits. Original photos stay untouched.</p>
    </div>
  </aside>
</main>
<script type="application/json" id="face-data">__FACE_PAYLOAD__</script>
<script>
(() => {
  'use strict';
  const payload = JSON.parse(document.getElementById('face-data').textContent);
  const ids = payload.landmark_ids;
  const views = new Set(['unknown','front','front-three-quarter','profile','rear-three-quarter','rear']);
  const source = payload.images;
  const storageKey = 'kumiki-face-annotation-v1:' + source.map(p =>
    p.image_id + ':' + p.sha256 + ':' + p.width + 'x' + p.height).join('|');
  const el = id => document.getElementById(id);
  const dom = {photos:el('photos'), landmarks:el('landmarks'), image:el('image'), overlay:el('overlay'),
    viewer:el('viewer'), stage:el('stage'), view:el('view'), outline:el('outline'),
    finish:el('finish-outline'), zoom:el('zoom'), zoomValue:el('zoom-value'),
    search:el('search'), visibility:el('point-visibility'), status:el('status'), progress:el('progress'),
    count:el('point-count'), undo:el('undo'), del:el('delete-point'), clear:el('clear-outline'),
    import:el('import'), importFile:el('import-file'), export:el('export')};
  const clone = value => JSON.parse(JSON.stringify(value));
  const initial = source.map(p => ({view:p.view, points:Object.fromEntries(p.landmarks.map(l => [l.landmark_id, l])),
    outline:p.outline_px ? clone(p.outline_px) : []}));
  let state = clone(initial);
  let active = 0;
  let selected = ids[0];
  let mode = 'point';
  let zoom = 1;
  const history = [];
  function announce(message, isError=false) {
    dom.status.textContent = message;
    dom.status.classList.toggle('error', isError);
  }
  function validPoint(point, image) {
    return Array.isArray(point) && point.length === 2 && point.every(v => Number.isFinite(v))
      && point[0] >= 0 && point[0] < image.width && point[1] >= 0 && point[1] < image.height;
  }
  function outlineArea2(outline) {
    return outline.reduce((sum, point, i) => {
      const next = outline[(i+1)%outline.length];
      return sum + point[0]*next[1] - next[0]*point[1];
    }, 0);
  }
  function validateImport(raw) {
    if (!raw || raw.schema_version !== 1 || !Array.isArray(raw.images) || raw.images.length > source.length)
      throw Error('Expected annotation JSON with schema_version=1 and an images array.');
    const next = clone(state);
    const seen = new Set();
    for (const row of raw.images) {
      const index = source.findIndex(photo => photo.image_id === row.image_id);
      if (index < 0 || seen.has(index)) throw Error('Unknown or duplicate image in imported annotations.');
      seen.add(index);
      if (!views.has(row.view)) throw Error('Invalid view in imported annotations.');
      if (!Array.isArray(row.landmarks) || row.landmarks.length > ids.length)
        throw Error('Invalid landmark list in imported annotations.');
      const points = {};
      for (const point of row.landmarks) {
        if (!ids.includes(point.landmark_id) || points[point.landmark_id])
          throw Error('Unknown or duplicate imported landmark.');
        if (!['manual','proposed'].includes(point.origin) ||
            !['visible','occluded','uncertain','not-visible'].includes(point.visibility))
          throw Error('Invalid imported landmark status.');
        if (row.view === 'rear' && point.visibility === 'visible')
          throw Error('A rear view cannot contain visible facial landmarks.');
        if (point.visibility === 'not-visible' ? point.xy_px !== null : !validPoint(point.xy_px, source[index]))
          throw Error('Imported landmark is outside the original photo or has invalid coordinates.');
        points[point.landmark_id] = clone(point);
      }
      const outline = row.outline_px == null ? [] : row.outline_px;
      if (!Array.isArray(outline) || (outline.length !== 0 && outline.length < 3) || outline.length > 2048
          || outline.some(point => !validPoint(point, source[index])))
        throw Error('Imported outline must contain 3–2048 in-bounds points.');
      if (outline.length >= 3 && Math.abs(outlineArea2(outline)) < 1)
        throw Error('Imported head outline has zero area.');
      next[index] = {view:row.view, points, outline:clone(outline)};
    }
    return next;
  }
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      const draft = JSON.parse(saved);
      state = validateImport(draft);
      if (Array.isArray(draft.partial_outlines) && draft.partial_outlines.length === source.length) {
        draft.partial_outlines.forEach((partial, index) => {
          if (Array.isArray(partial) && partial.length > 0 && partial.length < 3 &&
              partial.every(point => validPoint(point, source[index])))
            state[index].outline = clone(partial);
        });
      }
      announce('Restored private annotations from this browser.');
    }
  } catch (_error) { /* local file origin or storage restrictions: export still works */ }
  function save() {
    try {
      localStorage.setItem(storageKey, JSON.stringify({
        ...exportData(), partial_outlines:state.map(entry =>
          entry.outline.length > 0 && entry.outline.length < 3 ? entry.outline : null)
      }));
    } catch (_error) { /* download remains available */ }
  }
  function remember(index=active) {
    history.push(index < 0 ? {index:-1, before:clone(state)} : {index, before:clone(state[index])});
    if (history.length > 200) history.shift();
  }
  function undo() {
    const action = history.pop();
    if (!action) return announce('Nothing to undo.');
    if (action.index < 0) state = action.before;
    else { state[action.index] = action.before; active = action.index; }
    save(); render(true); announce('Last edit undone.');
  }
  function exportData() {
    return {schema_version:1, images:source.map((photo, index) => ({
      image_id:photo.image_id, view:state[index].view,
      landmarks:Object.values(state[index].points).sort((a,b) => a.landmark_id.localeCompare(b.landmark_id)),
      outline_px:state[index].outline.length >= 3 ? state[index].outline : null
    }))};
  }
  function downloadJson() {
    if (state.some(entry => entry.outline.length > 0 && entry.outline.length < 3))
      return announce('Finish or clear the incomplete outline before exporting.', true);
    if (state.some(entry => entry.outline.length >= 3 && Math.abs(outlineArea2(entry.outline)) < 1))
      return announce('A head outline has zero area. Undo recent vertices or clear it.', true);
    const blob = new Blob([JSON.stringify(exportData(), null, 2) + '\n'], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url; link.download = 'statue-face-annotations.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    announce('Annotation JSON downloaded. Keep it beside your private photos.');
  }
  function chooseImage(index) {
    if (index < 0 || index >= source.length) return;
    if (active !== index) { active = index; zoom = 1; dom.zoom.value = '1'; mode = 'point'; }
    render(true);
    announce('Click the image to place ' + selected.replaceAll('-', ' ') + '.');
  }
  function displayWidth() {
    const photo = source[active];
    const availableW = Math.max(150, dom.viewer.clientWidth - 36);
    const availableH = Math.max(160, dom.viewer.clientHeight - 36);
    return Math.max(100, Math.min(availableW/photo.width, availableH/photo.height) * photo.width * zoom);
  }
  function fitStage() {
    const width = displayWidth();
    const photo = source[active];
    dom.stage.style.width = width + 'px';
    dom.stage.style.height = width * photo.height/photo.width + 'px';
    dom.zoomValue.textContent = Math.round(zoom*100) + '%';
  }
  function renderPhotos() {
    if (dom.photos.children.length === 0) {
      source.forEach((photo, index) => {
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'photo-item';
        button.setAttribute('aria-label', 'Photo ' + (index+1) + ': ' + photo.image_id);
        const thumbnail = document.createElement('img'); thumbnail.src = photo.preview; thumbnail.alt = '';
        const name = document.createElement('span'); name.className = 'photo-name'; name.textContent = photo.image_id;
        const view = document.createElement('span'); view.className = 'photo-view';
        button.append(thumbnail, name, view);
        button.addEventListener('click', () => chooseImage(index));
        dom.photos.append(button);
      });
    }
    [...dom.photos.children].forEach((button, index) => {
      button.setAttribute('aria-current', index === active ? 'true' : 'false');
      button.querySelector('.photo-view').textContent = state[index].view.replaceAll('-', ' ');
    });
  }
  function renderLandmarks() {
    if (dom.landmarks.children.length === 0) {
      ids.forEach(name => {
        const button = document.createElement('button'); button.type = 'button';
        button.className = 'landmark-btn'; button.dataset.landmarkId = name;
        const label = document.createElement('span'); label.textContent = name.replaceAll('-', ' ');
        const status = document.createElement('span');
        button.append(label, status);
        button.addEventListener('click', () => {
          selected = name; mode = 'point'; render(false);
          announce('Click the photo to place ' + name.replaceAll('-', ' ') + '.');
        });
        dom.landmarks.append(button);
      });
    }
    const query = dom.search.value.trim().toLowerCase();
    let count = 0;
    [...dom.landmarks.children].forEach(button => {
      const name = button.dataset.landmarkId;
      button.hidden = !name.includes(query);
      if (button.hidden) return;
      const point = state[active].points[name];
      button.setAttribute('aria-pressed', selected === name ? 'true' : 'false');
      button.setAttribute('aria-label', name.replaceAll('-', ' ') + (point ? ', placed' : ', not placed'));
      const status = button.lastElementChild;
      status.textContent = point ? (point.origin === 'proposed' ? '◇' : '✓') : '·';
      status.className = point ? (point.origin === 'proposed' ? 'suggested' : 'set') : '';
      count++;
    });
    const placed = Object.values(state[active].points).filter(p => p.visibility === 'visible').length;
    dom.count.textContent = placed + ' placed · ' + count + ' shown';
    dom.visibility.value = state[active].points[selected]?.visibility || 'unmarked';
    dom.del.disabled = !state[active].points[selected];
    dom.del.title = dom.del.disabled ? 'Place the selected point before removing it' : 'Remove ' + selected;
  }
  function renderOverlay() {
    const photo = source[active], entry = state[active];
    const size = Math.max(5, photo.width * 0.010);
    dom.overlay.setAttribute('viewBox', `0 0 ${photo.width} ${photo.height}`);
    const segments = [];
    if (entry.outline.length > 0) {
      const points = entry.outline.map(p => p.join(',')).join(' ');
      const tag = entry.outline.length >= 3 && mode !== 'outline' ? 'polygon' : 'polyline';
      segments.push(`<${tag} points="${points}" fill="none" stroke="#67e4d5" stroke-width="${size/2}"/>`);
      for (const [x,y] of entry.outline)
        segments.push(`<circle cx="${x}" cy="${y}" r="${size/2}" fill="#67e4d5"/>`);
    }
    for (const point of Object.values(entry.points)) {
      if (!point.xy_px || point.visibility === 'not-visible') continue;
      const [x,y] = point.xy_px;
      const color = point.origin === 'proposed' ? '#efbd72' : '#6de1ce';
      const radius = point.landmark_id === selected ? size*1.0 : size*.7;
      segments.push(`<circle cx="${x}" cy="${y}" r="${radius}" stroke="#10212a" stroke-width="2" fill="${color}"/>`);
      if (point.landmark_id === selected)
        segments.push(`<circle cx="${x}" cy="${y}" r="${size*1.55}" fill="none" stroke="${color}" stroke-width="2"/>`);
    }
    dom.overlay.innerHTML = segments.join('');
  }
  function render(forceImage=false) {
    const photo = source[active];
    if (forceImage || dom.image.dataset.imageId !== photo.image_id) {
      dom.image.src = photo.preview; dom.image.dataset.imageId = photo.image_id;
      dom.image.onload = fitStage;
    }
    dom.image.alt = photo.image_id + ', ' + state[active].view.replaceAll('-', ' ') + ' view';
    dom.progress.textContent = 'Photo ' + (active+1) + ' of ' + source.length + ' · offline · private';
    dom.view.value = state[active].view;
    dom.outline.classList.toggle('is-active', mode === 'outline');
    dom.outline.setAttribute('aria-pressed', mode === 'outline' ? 'true' : 'false');
    dom.finish.disabled = mode !== 'outline' || state[active].outline.length < 3 ||
      Math.abs(outlineArea2(state[active].outline)) < 1;
    dom.clear.disabled = state[active].outline.length === 0;
    dom.undo.disabled = history.length === 0;
    renderPhotos(); renderLandmarks(); renderOverlay(); fitStage();
  }
  function toggleOutline() {
    mode = mode === 'outline' ? 'point' : 'outline';
    render(false);
    announce(mode === 'outline' ? 'Click around the head; finish with Enter (3+ points).'
      : 'Selected landmark: ' + selected.replaceAll('-', ' ') + '.');
  }
  function finishOutline() {
    if (state[active].outline.length < 3)
      return announce('Add at least three head-outline vertices.', true);
    if (Math.abs(outlineArea2(state[active].outline)) < 1)
      return announce('Head outline has zero area; undo or add a non-collinear vertex.', true);
    mode = 'point'; render(false); announce('Head outline ready. You can still add or clear vertices.');
  }
  function handlePlacement(event) {
    const rect = dom.stage.getBoundingClientRect(), photo = source[active];
    if (!rect.width || !rect.height) return;
    const x = Math.min(photo.width-0.001, Math.max(0, (event.clientX-rect.left)*photo.width/rect.width));
    const y = Math.min(photo.height-0.001, Math.max(0, (event.clientY-rect.top)*photo.height/rect.height));
    if (mode === 'outline') {
      remember(); state[active].outline.push([Math.round(x*10)/10, Math.round(y*10)/10]);
      announce(state[active].outline.length + ' outline vertices. Press Enter to finish.');
    } else {
      if (state[active].view === 'rear') return announce('Rear views cannot have visible facial points.', true);
      remember(); state[active].points[selected] = {landmark_id:selected,
        xy_px:[Math.round(x*10)/10, Math.round(y*10)/10], origin:'manual', visibility:'visible'};
      const next = ids.find(name => !state[active].points[name]);
      if (next) selected = next;
      announce('Point placed. ' + (next ? 'Next: ' + next.replaceAll('-', ' ') : 'All points reviewed.'));
    }
    save(); render(false);
  }
  dom.stage.addEventListener('click', handlePlacement);
  dom.view.addEventListener('change', () => {
    remember(); state[active].view = dom.view.value;
    if (state[active].view === 'rear') {
      for (const [name, point] of Object.entries(state[active].points))
        if (point.visibility === 'visible') delete state[active].points[name];
    }
    save(); render(false); announce('View label updated.');
  });
  dom.outline.addEventListener('click', toggleOutline);
  dom.finish.addEventListener('click', finishOutline);
  dom.zoom.addEventListener('input', () => {
    zoom = Number(dom.zoom.value); fitStage();
    dom.viewer.scrollLeft = (dom.viewer.scrollWidth-dom.viewer.clientWidth)/2;
    dom.viewer.scrollTop = (dom.viewer.scrollHeight-dom.viewer.clientHeight)/2;
  });
  dom.undo.addEventListener('click', undo);
  dom.search.addEventListener('input', renderLandmarks);
  dom.visibility.addEventListener('change', () => {
    const choice = dom.visibility.value;
    const current = state[active].points[selected];
    if (choice === 'visible' && state[active].view === 'rear') {
      render(false); return announce('Rear views cannot have visible facial points.', true);
    }
    if (choice !== 'unmarked' && choice !== 'not-visible' && !current?.xy_px) {
      render(false); return announce('Click a point on the photo before choosing this status.', true);
    }
    if (choice === (current?.visibility || 'unmarked')) return;
    remember();
    if (choice === 'unmarked') delete state[active].points[selected];
    else if (choice === 'not-visible') state[active].points[selected] = {
      landmark_id:selected, xy_px:null, origin:'manual', visibility:choice
    };
    else state[active].points[selected] = {...current, origin:'manual', visibility:choice};
    save(); render(false); announce('Selected feature marked ' + choice.replaceAll('-', ' ') + '.');
  });
  dom.del.addEventListener('click', () => {
    if (!state[active].points[selected]) return;
    remember(); delete state[active].points[selected]; save(); render(false);
    announce('Selected point removed. Undo restores it.');
  });
  dom.clear.addEventListener('click', () => {
    if (!state[active].outline.length) return;
    remember(); state[active].outline = []; save(); render(false);
    announce('Outline cleared. Undo restores it.');
  });
  dom.export.addEventListener('click', downloadJson);
  dom.import.addEventListener('click', () => dom.importFile.click());
  dom.importFile.addEventListener('change', async () => {
    const file = dom.importFile.files?.[0];
    if (!file) return;
    try {
      if (file.size > 2*1024*1024) throw Error('Annotation JSON exceeds 2 MiB.');
      const imported = validateImport(JSON.parse(await file.text()));
      remember(-1); state = imported; save(); render(true);
      announce('Imported annotations. Undo restores your previous work.');
    } catch (error) { announce('Import failed: ' + error.message, true); }
    dom.importFile.value = '';
  });
  document.addEventListener('keydown', event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
      event.preventDefault(); downloadJson(); return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault(); undo(); return;
    }
    if (['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName) || event.altKey || event.metaKey || event.ctrlKey)
      return;
    if (event.key === 'ArrowRight') { event.preventDefault(); chooseImage(active+1); }
    else if (event.key === 'ArrowLeft') { event.preventDefault(); chooseImage(active-1); }
    else if (event.key.toLowerCase() === 'o') { event.preventDefault(); toggleOutline(); }
    else if (event.key === 'Enter' && mode === 'outline') { event.preventDefault(); finishOutline(); }
    else if (event.key === 'Escape' && mode === 'outline') { mode='point'; render(false); }
    else if (event.key === 'Delete' || event.key === 'Backspace') { event.preventDefault(); dom.del.click(); }
  });
  window.addEventListener('resize', fitStage);
  render(true);
})();
</script>
</body>
</html>'''
