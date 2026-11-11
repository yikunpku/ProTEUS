'use strict';

const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const filmDetails = document.querySelector('#film');
const film = document.querySelector('#project-film');
document.querySelectorAll('[data-open-film]').forEach(link => {
  link.addEventListener('click', () => { filmDetails.open = true; });
});
function revealFilm() {
  if (location.hash === '#film') filmDetails.open = true;
}
window.addEventListener('hashchange', revealFilm);
revealFilm();
filmDetails.addEventListener('toggle', () => { if (!filmDetails.open) film.pause(); });
let pendingSeek = null;
document.querySelectorAll('[data-seek]').forEach(button => {
  button.addEventListener('click', () => {
    if (pendingSeek) film.removeEventListener('loadedmetadata', pendingSeek);
    const start = () => {
      pendingSeek = null;
      if (!filmDetails.open) return;
      film.currentTime = Number(button.dataset.seek);
      film.play().catch(() => {});
    };
    if (film.readyState >= 1) start();
    else {
      pendingSeek = start;
      film.addEventListener('loadedmetadata', start, { once: true });
      film.load();
      // Start within the click gesture; Safari may reject an unmuted play()
      // first issued by the later loadedmetadata event.
      film.play().catch(() => {});
    }
    film.focus({ preventScroll: true });
  });
});

const videos = [...document.querySelectorAll('.loop-video')];
const visibleVideos = new Set();
function syncVideo(video) {
  const allowed = visibleVideos.has(video) && !document.hidden && !video.closest('[hidden]') &&
    !video.dataset.userPaused && (!reducedMotion.matches || video.dataset.userPlaying === 'true');
  if (allowed) video.play().catch(() => {});
  else video.pause();
}
const motionObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) visibleVideos.add(entry.target);
    else visibleVideos.delete(entry.target);
    syncVideo(entry.target);
  });
}, { threshold: .2 });
videos.forEach(video => {
  const button = document.querySelector(`[data-video="${video.id}"]`);
  function label() {
    button.textContent = video.paused ? 'Play' : 'Pause';
    button.setAttribute('aria-label', `${video.paused ? 'Play' : 'Pause'} animation`);
  }
  video.addEventListener('play', label);
  video.addEventListener('pause', label);
  button.addEventListener('click', () => {
    if (video.paused) {
      delete video.dataset.userPaused;
      video.dataset.userPlaying = 'true';
      video.play().catch(() => {});
    } else {
      video.dataset.userPaused = 'true';
      delete video.dataset.userPlaying;
      video.pause();
    }
  });
  label();
  motionObserver.observe(video);
});
document.addEventListener('visibilitychange', () => videos.forEach(syncVideo));

function selectTab(button, selector, panel) {
  document.querySelectorAll(selector).forEach(tab => {
    const selected = tab === button;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  panel.setAttribute('aria-labelledby', button.id);
}
const methods = {
  world: { src: 'world-model-loop', title: 'Mutation-resolved world model with a closed wet-lab loop' },
  modalities: { src: 'architecture', title: 'Three modalities, one joint embedding space' },
  attention: { src: 'attention-comparison', title: 'Mutation-resolved attention: ProTEUS and ESM-2' },
  search: { src: 'guided-exploration', title: 'Mutation-guided exploration through three real SacB mutation steps' }
};
const methodVideo = document.querySelector('#method-video');
const methodButtons = [...document.querySelectorAll('[data-method]')];
methodVideo.loop = false;
function showMethod(button, userInitiated = false) {
  const method = methods[button.dataset.method];
  selectTab(button, '[data-method]', document.querySelector('#method-panel'));
  methodVideo.poster = `assets/video/${method.src}-poster.webp?v=layout-v12`;
  methodVideo.querySelector('source').src = `assets/video/${method.src}.mp4?v=layout-v12`;
  document.querySelector('#method-enlarge').href = `assets/video/${method.src}.mp4?v=layout-v12`;
  methodVideo.setAttribute('aria-label', method.title);
  if (userInitiated) {
    delete methodVideo.dataset.userPaused;
    methodVideo.dataset.userPlaying = 'true';
  }
  // Keep the user's play/pause choice across automatic transitions. Initial
  // playback still honors reduced motion and viewport visibility.
  methodVideo.load();
  syncVideo(methodVideo);
}
methodButtons.forEach(button => button.addEventListener('click', () => showMethod(button, true)));
methodVideo.addEventListener('ended', () => {
  if (!methodVideo.ended) return;
  const current = methodButtons.findIndex(button => button.getAttribute('aria-selected') === 'true');
  showMethod(methodButtons[(current + 1) % methodButtons.length]);
});
document.querySelectorAll('[data-study]').forEach(button => {
  button.addEventListener('click', () => {
    const { study, view } = button.dataset;
    ['activity', 'function'].forEach(name => {
      document.getElementById(`${study}-${name}-panel`).hidden = name !== view;
    });
    selectTab(button, `[data-study="${study}"]`, document.getElementById(`${study}-${view}-panel`));
    videos.forEach(syncVideo);
  });
});
document.querySelectorAll('[role="tablist"]').forEach(list => {
  list.addEventListener('keydown', event => {
    const tabs = [...list.querySelectorAll('[role="tab"]')];
    const at = tabs.indexOf(document.activeElement);
    if (at < 0) return;
    let next;
    if (event.key === 'ArrowRight') next = (at + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (at + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault();
    tabs[next].focus();
    tabs[next].click();
  });
});

const chartStates = new Map();
function createChart(prefix, name) {
  const canvas = document.querySelector(`#${prefix}-chart`);
  const ctx = canvas.getContext('2d');
  const state = { canvas, name, frame: 0, width: 0, height: 0, progress: 1 };
  function draw(progress = state.progress) {
    state.progress = progress;
    const { width, height } = state;
    if (!width) return;
    ctx.fillStyle = window.ProTEUSCharts.background;
    ctx.fillRect(0, 0, width, height);
    const box = width < 600 ? [0, 0, width, height] : [35, 30, width - 70, height - 70];
    window.ProTEUSCharts.drawChart(ctx, state.name, box, progress);
  }
  function resize() {
    const cssWidth = canvas.parentElement.clientWidth;
    if (!cssWidth) return;
    const size = window.ProTEUSCharts.layoutForWidth(cssWidth,state.name);
    state.width = size.width;
    state.height = size.height;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(size.height * dpr);
    canvas.style.height = `${cssWidth * size.height / size.width}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.proteusCssUnit = cssWidth / (size.width * dpr);
    draw();
  }
  state.resize = resize;
  state.animate = () => {
    cancelAnimationFrame(state.frame);
    if (reducedMotion.matches) { draw(1); return; }
    const start = performance.now();
    const tick = now => {
      const p = Math.min(1, (now - start) / 1000);
      draw(p * p * (3 - 2 * p));
      if (p < 1) state.frame = requestAnimationFrame(tick);
    };
    state.frame = requestAnimationFrame(tick);
  };
  state.finish = () => { cancelAnimationFrame(state.frame); draw(1); };
  new ResizeObserver(resize).observe(canvas.parentElement);
  resize();
  const observer = new IntersectionObserver(entries => {
    if (entries.some(entry => entry.isIntersecting)) { state.animate(); observer.disconnect(); }
  }, { threshold: .15 });
  observer.observe(canvas);
  document.querySelector(`[data-replay="${prefix}"]`).addEventListener('click', state.animate);
  chartStates.set(prefix, state);
  return state;
}
createChart('scaling', 'Fig2C');
const predictionChart = createChart('prediction', 'Fig2E');
const benchmarkChart = createChart('benchmark', 'Fig3D');
const predictions = {
  "zero": {
    "chart": "Fig2E",
    "description": "Zero-shot prediction and first-round selection.",
    "note": "19 prediction assays; 13 single-mutant selection assays. No target fitness labels. Mean ± SEM. MULTI-evolve: the paper’s rank-fusion adaptation (Figures 2E–F)."
  },
  "assays": {
    "chart": "SelectionQuality",
    "description": "First-round selection on 13 held-out single-mutant assays.",
    "note": "Spearman, AUROC, AUPR and hit fraction; mean ± SEM. Hit threshold: assay mean. Comparator: MULTI-evolve rank fusion."
  },
  "selection": {
    "chart": "Fig2F_selection",
    "description": "Mean top-10 hit fraction: 75.4%.",
    "note": "13 single-mutant assays; mean ± SEM. Hits meet or exceed the assay mean, not necessarily WT. At most one substitution per residue position. MULTI-evolve: the paper’s zero-shot rank-fusion adaptation."
  },
  "guided": {
    "chart": "Fig3G",
    "description": "Guided evolution on 17 multi-mutant DMS datasets, compared with EVOLVEpro and MULTI-evolve.",
    "note": "In silico · 30 initial variants + 5 rounds of 30 · matched libraries and budgets. Normalized best fitness, mean ± SEM."
  }
};
document.querySelectorAll('[data-prediction]').forEach(button => {
  button.addEventListener('click', () => {
    const item = predictions[button.dataset.prediction];
    selectTab(button, '[data-prediction]', document.querySelector('#prediction-panel'));
    ['description', 'note'].forEach(key => { document.querySelector('#prediction-' + key).textContent = item[key]; });
    document.querySelector('#prediction-references').hidden = button.dataset.prediction !== 'guided';
    predictionChart.name = item.chart;
    predictionChart.resize();
    predictionChart.canvas.setAttribute('aria-label', item.description + ' ' + item.note);
    predictionChart.canvas.textContent = item.description + ' ' + item.note;
    predictionChart.animate();
  });
});
const benchmarks = {
  speed: { chart: 'Fig3D', title: '7.8–10.8× faster', description: 'Than AlphaFold 3 across five CASP15 atom-count bins on an NVIDIA RTX A6000.', note: 'Projected structures passed all seven specified validity checks in the reported benchmarks.' },
  higher: { chart: 'Fig3F', title: '0.448 vs 0.295', description: 'Mean Spearman, compared with the MULTI-evolve one-hot predictor, across 40 dataset–mutation-order comparisons.', note: 'Train on variants with at most K substitutions; test on variants with more than K. Figure 3F.' },
  evolution: { chart: 'Fig3G', title: 'Five model-guided rounds', description: 'Compared with EVOLVEpro and MULTI-evolve, ProTEUS reached the highest mean normalized best fitness in each round.', note: 'In silico, across 17 DMS datasets: 30 initial variants + five rounds of 30, with matched candidate libraries and nomination budgets. Figure 3G.' }
};
document.querySelectorAll('[data-benchmark]').forEach(button => {
  button.addEventListener('click', () => {
    const item = benchmarks[button.dataset.benchmark];
    document.querySelector('#benchmark-panel').dataset.component = button.dataset.benchmark === 'speed' ? 'structure' : 'prediction';
    selectTab(button, '[data-benchmark]', document.querySelector('#benchmark-panel'));
    ['title', 'description', 'note'].forEach(key => { document.querySelector('#benchmark-' + key).textContent = item[key]; });
    benchmarkChart.name = item.chart;
    benchmarkChart.resize();
    benchmarkChart.canvas.setAttribute('aria-label', item.title + '. ' + item.description);
    benchmarkChart.animate();
  });
});
reducedMotion.addEventListener('change', () => {
  videos.forEach(video => { delete video.dataset.userPlaying; syncVideo(video); });
  chartStates.forEach(state => state.finish());
});

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex');
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(citation.textContent);
    status.textContent = 'BibTeX copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Use Copy on your device, or press Ctrl+C / ⌘C.';
  }
});
const legacySections = { overview: 'model', benchmarks: 'structure', scaling: 'prediction', scientists: 'experiments' };
const target = legacySections[location.hash.slice(1)];
if (target) {
  history.replaceState(null, '', '#' + target);
  document.getElementById(target).scrollIntoView();
}
const links = [...document.querySelectorAll('.site-header nav a')];
const visibleSections = new Set();
const sectionObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) visibleSections.add(entry.target.id);
    else visibleSections.delete(entry.target.id);
  });
  const active = links.find(link => visibleSections.has(link.hash.slice(1)));
  links.forEach(link => {
    if (link === active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}, { rootMargin: '-15% 0px -60% 0px' });
document.querySelectorAll('.research-section').forEach(section => sectionObserver.observe(section));
