#!/usr/bin/env node
'use strict';
/**
 * Re-render the paper's quantitative results as native Canvas animation.
 *
 * Usage:
 *   node tools/video/render_charts.cjs
 *   node tools/video/render_charts.cjs --preview
 *   node tools/video/render_charts.cjs --only sacb-activity,aldh-activity
 *   node tools/video/render_charts.cjs --refresh-data --source-root /path/to/scripts
 *
 * Requires @napi-rs/canvas, FFmpeg, and Georgia. CANVAS_MODULE, GEORGIA_TTF,
 * GEORGIA_BOLD_TTF, and FFMPEG may override local dependency locations.
 * Five benchmark clips use the website's numerical data and chart renderer.
 * Enzyme data are a checked snapshot of the supplied manuscript CSV files.
 * No rasterized paper panels, generated molecular images, or invented points.
 */
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const vm = require('node:vm');
const crypto = require('node:crypto');
const {spawn} = require('node:child_process');
const {once} = require('node:events');

const ROOT = path.resolve(__dirname, '../..');
const SNAPSHOT = path.join(__dirname, 'chart-data.json');
const ENGINE = path.join(ROOT, 'docs/chart-engine.js');
const W = 1920, H = 1080, LW = 1280, LH = 720, FPS = 24, SECONDS = 6;
const REVEAL_SECONDS = 3;
const ARGS = process.argv.slice(2);
function option(name, fallback) {
  const at = ARGS.indexOf(name);
  if (at < 0) return fallback;
  if (!ARGS[at + 1] || ARGS[at + 1].startsWith('--')) throw Error(`Missing ${name} value`);
  return ARGS[at + 1];
}
const OUT = path.resolve(option('--out', path.join(ROOT, 'docs/assets/video')));
const SOURCE = path.resolve(option('--source-root', path.join(os.homedir(), 'Dropbox/PROTEUS/Submission_V2/main_figure/scripts')));
const PREVIEW = ARGS.includes('--preview');
const ONLY = option('--only', '').split(',').filter(Boolean);
const sha = input => crypto.createHash('sha256').update(input).digest('hex');
const clamp = x => Math.max(0, Math.min(1, x));
const ease = x => { const p = clamp(x); return p * p * (3 - 2 * p); };
const C = {ink:'#293E50', muted:'#607381', blue:'#4383AB', axis:'#9BABBA', grid:'#E4EAF0'};
const STROKES = ['#AC936B', '#529C96', '#4383AB', '#85919B', '#0A4363'];
const FILLS = ['#F7EED8', '#E8F4EF', '#8BBBD5', '#DADEE1', '#6D9EC6'];

function requireCanvas() {
  const candidates = [process.env.CANVAS_MODULE, '@napi-rs/canvas',
    path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@napi-rs/canvas')].filter(Boolean);
  for (const modulePath of candidates) {
    try { return require(modulePath); } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND') throw error;
    }
  }
  throw Error('Install @napi-rs/canvas or set CANVAS_MODULE to its location.');
}
const {createCanvas, GlobalFonts} = requireCanvas();
const fontRegular = process.env.GEORGIA_TTF || '/System/Library/Fonts/Supplemental/Georgia.ttf';
const fontBold = process.env.GEORGIA_BOLD_TTF || path.join(path.dirname(fontRegular), 'Georgia Bold.ttf');
for (const filename of [fontRegular, fontBold]) {
  if (!fs.existsSync(filename)) throw Error(`Georgia font missing: ${filename}`);
  if (!GlobalFonts.registerFromPath(filename, 'Georgia')) throw Error(`Cannot load font: ${filename}`);
}
const engineSource = fs.readFileSync(ENGINE, 'utf8');
const sandbox = {window:{}};
vm.runInNewContext(engineSource, sandbox, {filename:ENGINE});
const charts = sandbox.window.ProTEUSCharts;

// These scientific input CSVs contain no quoted delimiters. Reject them if
// that changes rather than silently reading a different table structure.
function csv(relative) {
  const content = fs.readFileSync(path.join(SOURCE, relative), 'utf8');
  const lines = content.trim().split(/\r?\n/);
  if (content.includes('"')) throw Error(`Quoted CSV requires a parser: ${relative}`);
  const keys = lines.shift().split(',');
  return lines.map(line => {
    const values = line.split(',');
    if (keys.length !== values.length) throw Error(`CSV column mismatch: ${relative}`);
    return Object.fromEntries(keys.map((key, i) => [key, values[i]]));
  });
}
function sourceInfo(relative) {
  const bytes = fs.readFileSync(path.join(SOURCE, relative));
  return {path:relative, sha256:sha(bytes), bytes:bytes.length};
}
function assertTopTen(raw, selected, rounds, name) {
  if (raw.length !== rounds * 30 || new Set(raw.map(r => r.variant)).size !== raw.length)
    throw Error(`${name}: expected ${rounds * 30} unique measured non-WT variants`);
  if (selected.length !== rounds * 10) throw Error(`${name}: unexpected displayed point count`);
  if (![...raw, ...selected].every(r => Number.isFinite(r.activity))) throw Error(`${name}: invalid activity`);
  for (let round = 0; round < rounds; round++) {
    const eligible = raw.filter(r => r.round <= round);
    const expected = eligible.slice().sort((a, b) => b.activity - a.activity).slice(0, 10);
    const actual = selected.filter(r => r.snapshot_round === round);
    if (eligible.length !== (round + 1) * 30 || actual.length !== 10) throw Error(`${name}: invalid round ${round}`);
    if (new Set(actual.map(r => r.variant)).size !== 10) throw Error(`${name}: repeated point within round`);
    for (const row of actual) {
      const original = expected.find(r => r.variant === row.variant);
      if (!original || original.activity !== row.activity || original.round !== row.source_round)
        throw Error(`${name}: selected point differs from raw table: ${row.variant}`);
      if (!Number.isFinite(row.plot_x) || Math.abs(row.plot_x - round) > .341)
        throw Error(`${name}: invalid horizontal packing`);
    }
  }
}
function refreshData() {
  const paths = {
    sacbRaw:'Figure4/data/tables/master_LSC.csv',
    sacbDots:'Figure4/generated/panel_activity.dots.json',
    sacbMethod:'Figure4/build_panel_activity.py',
    aldhRaw:'Figure5/data/fig3b_latent_ALDH_tsne_points.csv',
    aldhDots:'Figure5/data/cumulative_top10_ALDH.csv',
    aldhMethod:'Figure5/build_panel_b.py',
  };
  const sacbRaw = csv(paths.sacbRaw).filter(r => +r.n_mut > 0)
    .map(r => ({variant:r.variant, round:+r.round, activity:+r.activity}));
  const sacb = JSON.parse(fs.readFileSync(path.join(SOURCE, paths.sacbDots), 'utf8'))
    .map(r => ({snapshot_round:r.snapshot_round, source_round:r.source_round,
      variant:r.variant, activity:r.activity, plot_x:r.x}));
  const aldhRaw = csv(paths.aldhRaw).filter(r => !['WT','wild_type'].includes(r.variant))
    .map(r => ({variant:r.variant, round:+r.round, activity:+r.activity}));
  const aldh = csv(paths.aldhDots).map(r => ({snapshot_round:+r.display_round,
    source_round:+r.assay_round, variant:r.variant, activity:+r.activity, plot_x:+r.plot_x}));
  assertTopTen(sacbRaw, sacb, 3, 'SacB');
  assertTopTen(aldhRaw, aldh, 5, 'ALDH');
  const snapshot = {
    description:'Exact cumulative top-10 activity observations and original horizontal packing from the supplied manuscript sources.',
    methods:{selection:'At snapshot r, take the 10 highest activities among non-WT variants measured through round r; 30 candidates assayed per round.',
      repeated_variants:'Earlier variants may recur in later cumulative snapshots; snapshot dots are not independent discoveries.',
      density:'Gaussian KDE, covariance factor 0.35, unbiased sample variance, 256 points within observed min/max, maximum halfwidth 0.34 categories. Descriptive density, not uncertainty.',
      jitter:'Original deterministic horizontal packing only; no vertical jitter.',
      baseline:'Activity is fold of wild type (WT = 1).'},
    sources:Object.fromEntries(Object.entries(paths).map(([key, relative]) => [key, sourceInfo(relative)])),
    sacb:{rounds:3, assayed:90, raw:sacbRaw, points:sacb},
    aldh:{rounds:5, assayed:150, raw:aldhRaw, points:aldh},
  };
  fs.writeFileSync(SNAPSHOT, JSON.stringify(snapshot, null, 2) + '\n');
}
if (ARGS.includes('--refresh-data') || !fs.existsSync(SNAPSHOT)) refreshData();
const data = JSON.parse(fs.readFileSync(SNAPSHOT, 'utf8'));
assertTopTen(data.sacb.raw, data.sacb.points, 3, 'SacB');
assertTopTen(data.aldh.raw, data.aldh.points, 5, 'ALDH');

function label(ctx, value, x, y, size = 28, color = C.ink, align = 'left', bold = false) {
  ctx.save();
  ctx.font = `${bold ? 700 : 400} ${size}px Georgia`;
  ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillStyle = color;
  ctx.fillText(String(value), x, y); ctx.restore();
}
function line(ctx, points, color, width = 2, dash = []) {
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
  ctx.beginPath();
  points.forEach(([x,y], i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
  ctx.stroke(); ctx.restore();
}
function kde(values) {
  const mean = values.reduce((a,v) => a + v, 0) / values.length;
  const variance = values.reduce((a,v) => a + (v - mean)**2, 0) / (values.length - 1);
  const bandwidth = .35 * Math.sqrt(variance);
  if (!(bandwidth > 0)) throw Error('Degenerate density');
  const lo = Math.min(...values), hi = Math.max(...values);
  const grid = Array.from({length:256}, (_, i) => {
    const y = lo + (hi - lo) * i / 255;
    const density = values.reduce((a,v) => a + Math.exp(-.5 * ((y-v)/bandwidth)**2), 0)
      / (values.length * bandwidth * Math.sqrt(2*Math.PI));
    return [y, density];
  });
  const peak = Math.max(...grid.map(r => r[1]));
  return grid.map(([y, density]) => [y, .34 * density / peak]);
}
const densityCache = {};
function activity(ctx, key, progress) {
  const source = data[key], rounds = source.rounds, maxY = key === 'sacb' ? 16 : 8;
  const bounds = [124, 117, 1084, 447];
  const xm = x => bounds[0] + bounds[2] * (x + .5) / rounds;
  const ym = y => bounds[1] + bounds[3] * (1 - y / maxY);
  label(ctx, 'Cumulative top 10 measured variants', 52, 35, 27, C.muted);
  label(ctx, 'Activity relative to WT', bounds[0], 80, 29);
  const ticks = key === 'sacb' ? [0,5,10,15] : [0,2,4,6,8];
  for (const value of ticks) {
    line(ctx, [[bounds[0],ym(value)],[bounds[0]+bounds[2],ym(value)]], C.grid, 1);
    label(ctx, value, bounds[0]-22, ym(value), 27, C.muted, 'right');
  }
  line(ctx, [[bounds[0],bounds[1]],[bounds[0],ym(0)],[bounds[0]+bounds[2],ym(0)]], C.axis, 2);
  line(ctx, [[bounds[0],ym(1)],[bounds[0]+bounds[2],ym(1)]], C.muted, 1.5, [8,7]);
  label(ctx, 'WT', bounds[0]-22, ym(1), 21, C.muted, 'right');
  for (let round = 0; round < rounds; round++) {
    const points = source.points.filter(r => r.snapshot_round === round);
    const values = points.map(r => r.activity);
    const cacheKey = `${key}-${round}`;
    const shape = densityCache[cacheKey] ||= kde(values);
    const alpha = ease(progress * rounds - round);
    const stroke = STROKES[round], fill = FILLS[round];
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.beginPath();
    shape.forEach(([y, width], i) => i ? ctx.lineTo(xm(round-width),ym(y)) : ctx.moveTo(xm(round-width),ym(y)));
    [...shape].reverse().forEach(([y,width]) => ctx.lineTo(xm(round+width),ym(y)));
    ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke();
    for (const point of points) {
      ctx.beginPath(); ctx.arc(xm(point.plot_x),ym(point.activity),6,0,2*Math.PI);
      ctx.fillStyle = stroke; ctx.fill(); ctx.strokeStyle = '#FFFFFF'; ctx.lineWidth = 1.1; ctx.stroke();
    }
    const best = Math.max(...values);
    label(ctx, `${best.toFixed(2)}×`, xm(round), ym(best)-27, 30, stroke, 'center', true);
    ctx.restore();
    label(ctx, round, xm(round), ym(0)+33, 29, C.ink, 'center');
    label(ctx, `${30*(round+1)} assayed`, xm(round), ym(0)+65, 22, C.muted, 'center');
  }
  label(ctx, 'Round', bounds[0]+bounds[2]/2, 663, 28, C.muted, 'center');
  label(ctx, 'Earlier variants may reappear in subsequent cumulative snapshots.', LW/2, 701, 22, C.muted, 'center');
}

const CLIPS = [
  {name:'scaling', chart:'Fig2C', source:'Figure 2C', semantics:'Eight measured scaling markers; original SVG coordinates and fitted exponent. Both axes explicitly mark logarithmic scales where used.'},
  {name:'zero-shot', chart:'Fig2E', source:'Figures 2E and 2F', semantics:'Zero-shot fitness prediction: all 15 model means/SEM across 19 excluded assays. First-round selection on 13 single-mutant assays: Spearman, AUROC, AUPR and top-K hit fractions; mean/SEM. MULTI-evolve rank-fusion adaptation. Figure 2F axis ranges and broken baselines; no significance stars. Hits are at or above assay mean, not WT.'},
  {name:'structure-speed', chart:'Fig3D', source:'Figure 3D', semantics:'Mean inference time in five atom-count bins; log time axis, exact measured ratios.'},
  {name:'higher-order', chart:'Fig3F', source:'Figure 3F', semantics:'All 40 protein-phenotype-K comparisons, K=1/2/3; means, SEM and individual values.'},
  {name:'closed-loop', chart:'Fig3G', source:'Figure 3G', semantics:'Simulated multi-mutant evolution on 17 datasets; normalized best fitness with SEM, 5 rounds and 4 methods.'},
  {name:'sacb-activity', activity:'sacb', source:'Figure 4C', semantics:'Measured SacB cumulative top 10 per round, 90 assayed variants, 3 snapshots, WT=1.'},
  {name:'aldh-activity', activity:'aldh', source:'Figure 5B', semantics:'Measured ALDH cumulative top 10 per round, 150 assayed variants, 5 snapshots, WT=1.'},
];
if (ONLY.some(name => !CLIPS.some(c => c.name === name))) throw Error(`Unknown clip in --only: ${ONLY.join(',')}`);
const canvas = createCanvas(W,H);
const ctx = canvas.getContext('2d');
function draw(clip, progress) {
  ctx.setTransform(1,0,0,1,0,0); ctx.globalAlpha = 1;
  ctx.fillStyle = '#FFFFFF'; ctx.fillRect(0,0,W,H);
  ctx.save(); ctx.scale(W/LW,H/LH);
  if (clip.name === 'scaling') {
    charts.drawChart(ctx,clip.chart,[20,30,900,660],progress);
    const a = ease((progress-.3)/.5);
    ctx.save(); ctx.globalAlpha=a;
    label(ctx,'80×',1000,165,67,'#3F7EAD');
    label(ctx,'training data',1000,224,24,C.muted);
    label(ctx,'56.3%',1000,347,62,'#3F7EAD');
    label(ctx,'lower test MSE',1000,406,24,C.muted);
    label(ctx,'0.262 to 0.115',1000,458,25,C.ink);
    ctx.restore();
  } else if (clip.chart) charts.drawChart(ctx,clip.chart,[35,30,1210,660],progress);
  else activity(ctx,clip.activity,progress);
  ctx.restore();
}
async function encode(clip) {
  draw(clip,1);
  fs.writeFileSync(path.join(OUT, `${clip.name}-poster.webp`), canvas.encodeSync('webp',92));
  if (PREVIEW) return;
  const output = path.join(OUT, `${clip.name}.mp4`);
  const encoder = spawn(process.env.FFMPEG || 'ffmpeg', ['-hide_banner','-loglevel','error','-y',
    '-f','rawvideo','-pixel_format','rgba','-video_size',`${W}x${H}`,'-framerate',String(FPS),'-i','pipe:0',
    '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
    '-metadata',`comment=ProTEUS ${clip.source}; native Canvas rendering from measured data`, output],
    {stdio:['pipe','ignore','pipe']});
  let errors = '';
  encoder.stderr.on('data', chunk => { errors += chunk.toString(); });
  const finished = new Promise((resolve,reject) => {
    encoder.once('error', reject);
    encoder.once('close', code => code === 0 ? resolve() : reject(Error(`FFmpeg failed (${code}): ${errors}`)));
  });
  // Attach immediately to avoid an unhandled rejection if FFmpeg is missing.
  finished.catch(() => {});
  encoder.stdin.on('error', () => {});
  for (let frame = 0; frame < FPS*SECONDS; frame++) {
    const progress = ease((frame/FPS)/REVEAL_SECONDS);
    draw(clip,progress);
    const rgba = ctx.getImageData(0,0,W,H).data;
    const buffer = Buffer.from(rgba.buffer,rgba.byteOffset,rgba.byteLength);
    if (!encoder.stdin.write(buffer)) {
      await Promise.race([once(encoder.stdin,'drain'),finished.then(() => {throw Error('Encoder closed early');})]);
    }
  }
  encoder.stdin.end();
  await finished;
  process.stdout.write(`${clip.name}: ${W}×${H}, ${FPS*SECONDS} frames, ${SECONDS}s\n`);
}
async function main() {
  fs.mkdirSync(OUT, {recursive:true});
  for (const clip of CLIPS.filter(c => !ONLY.length || ONLY.includes(c.name))) await encode(clip);
  const manifest = {
    description:'Original quantitative data re-rendered as Georgia Canvas graphics. No screenshot or previously encoded movie is an input.',
    width:W,height:H,fps:FPS,duration_seconds:SECONDS,reveal_seconds:REVEAL_SECONDS,
    animation:'Measured positions are revealed. Connecting lines and aggregate marks animate as visual transitions, not additional observations or simulated trajectories.',
    renderer:'tools/video/render_charts.cjs',chart_engine:'docs/chart-engine.js',
    chart_engine_sha256:sha(engineSource),
    chart_data_sha256:sha(engineSource.match(/\/\* DATA_START \*\/([\s\S]*?)\/\* DATA_END \*\//)[1]),
    activity_snapshot:'tools/video/chart-data.json',activity_snapshot_sha256:sha(fs.readFileSync(SNAPSHOT)),
    fonts:[path.basename(fontRegular),path.basename(fontBold)],
    clips:CLIPS.map(c => ({...c, video:`${c.name}.mp4`,poster:`${c.name}-poster.webp`,
      ...(fs.existsSync(path.join(OUT,`${c.name}.mp4`)) ? {sha256:sha(fs.readFileSync(path.join(OUT,`${c.name}.mp4`)))} : {})})),
  };
  fs.writeFileSync(path.join(OUT,'chart-manifest.json'),JSON.stringify(manifest,null,2)+'\n');
}
main().catch(error => { process.stderr.write(`${error.stack}\n`); process.exitCode = 1; });
