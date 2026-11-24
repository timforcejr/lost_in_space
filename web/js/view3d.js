// The interactive 3-d map (three.js): reference cloud, the two typed sentences, their words,
//       lit-up neighbors, layer trajectories, hover and pick.
// Verification: tests/smoke.mjs screenshots

import * as THREE from '../vendor/three/three.module.js';
import { OrbitControls } from '../vendor/three/OrbitControls.js';

const VERT = `
attribute float aSize; attribute float aAlpha; attribute float aShape;
varying vec3 vColor; varying float vAlpha; varying float vShape;
uniform float uScale; uniform float uPx;   // uPx = device pixels per CSS pixel, so points are the same size on HiDPI screens
void main() {
  vColor = color; vAlpha = aAlpha; vShape = aShape;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = clamp(aSize * uScale / -mv.z, 1.5, 30.0) * uPx;
  gl_Position = projectionMatrix * mv;
}`;
const FRAG = `
varying vec3 vColor; varying float vAlpha; varying float vShape;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float d;
  if (vShape < 0.5) d = length(p);                                  // circle
  else if (vShape < 1.5) d = max(abs(p.x), abs(p.y)) * 1.12;        // square
  else if (vShape < 2.5) d = max(abs(p.x) * 0.95 - p.y * 0.55, p.y) * 1.25 + 0.12;  // triangle
  else d = (abs(p.x) + abs(p.y)) * 0.92;                            // diamond
  float a = 1.0 - smoothstep(0.78, 1.0, d);
  if (a < 0.02 || vAlpha < 0.01) discard;
  gl_FragColor = vec4(vColor, a * vAlpha);
}`;

function glowTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d'), grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,255,255,1)'); grad.addColorStop(0.25, 'rgba(255,255,255,0.45)'); grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

export class View3D {
  constructor(container, { onHover = () => {}, onPick = () => {} } = {}) {
    this.el = container; this.onHover = onHover; this.onPick = onPick;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    container.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
    this.camera.position.set(5.2, 3.2, 7.4);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: 0.08, autoRotate: !matchMedia('(prefers-reduced-motion: reduce)').matches, autoRotateSpeed: 0.35, minDistance: 0.6, maxDistance: 40 });
    this.controls.addEventListener('start', () => { this.controls.autoRotate = false; });
    this.query = new THREE.Group(); this.scene.add(this.query);
    this.labels = []; this.labelLayer = container.querySelector('.labels');
    this.glow = glowTexture();
    this.anim = null; this.n = 0; this.camAnim = null;
    this.ray = new THREE.Raycaster(); this.ray.params.Points.threshold = 0.07;
    this._bind();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(t => this.frame(t));
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    if (this.points) this.points.material.uniforms.uScale.value = h * 0.9;
  }

  /** positions Float32Array(n*3); colors Float32Array(n*3); shapes Float32Array(n). Animates if a cloud exists. */
  setCloud(positions, colors, shapes) {
    const n = positions.length / 3;
    if (!this.points || this.n !== n) {
      if (this.points) this.scene.remove(this.points);
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(positions.slice(), 3));
      g.setAttribute('color', new THREE.BufferAttribute(colors.slice(), 3));
      g.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(n).fill(0.06), 1));
      g.setAttribute('aAlpha', new THREE.BufferAttribute(new Float32Array(n).fill(0.8), 1));
      g.setAttribute('aShape', new THREE.BufferAttribute(shapes.slice(), 1));
      const m = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, vertexColors: true, transparent: true, depthWrite: false,
        uniforms: { uScale: { value: this.el.clientHeight * 0.9 }, uPx: { value: this.renderer.getPixelRatio() } } });
      this.points = new THREE.Points(g, m); this.points.frustumCulled = false;
      this.scene.add(this.points); this.n = n;
    } else {
      this.anim = { from: this.points.geometry.attributes.position.array.slice(), to: positions.slice(), t0: performance.now(), ms: 900 };
      this.setColors(colors);
    }
    this.target = positions;
  }
  setColors(colors) { const a = this.points.geometry.attributes.color; a.array.set(colors); a.needsUpdate = true; }
  setEmphasis(sizes, alphas) {
    const g = this.points.geometry;
    g.attributes.aSize.array.set(sizes); g.attributes.aSize.needsUpdate = true;
    g.attributes.aAlpha.array.set(alphas); g.attributes.aAlpha.needsUpdate = true;
    this.alphas = alphas;
  }

  clearQuery() {
    this.query.traverse(o => { o.geometry?.dispose?.(); if (o.material && o.material.map !== this.glow) o.material.dispose?.(); });
    this.query.clear();
    this.labels.forEach(l => l.el.remove()); this.labels = [];
  }

  _label(text, pos, cls) {
    const el = document.createElement('div'); el.className = 'label ' + cls; el.textContent = text;
    this.labelLayer.append(el); this.labels.push({ el, pos: new THREE.Vector3(...pos) });
  }

  /**
   * sentences: [{key:'A'|'B', color:'#hex', pos:[x,y,z], words:[{text,pos,weight}], traj:[[x,y,z]...], links:[[x,y,z]...]}]
   * showWords / showTraj toggle the optional layers.
   */
  setQuery(sentences, { showWords = true, showTraj = true, layer = 12 } = {}) {
    this.clearQuery();
    const line = (pts, color, opacity, dashed = false) => {
      const g = new THREE.BufferGeometry().setFromPoints(pts.map(p => new THREE.Vector3(...p)));
      const m = dashed ? new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: 0.09, gapSize: 0.06 })
                       : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
      const l = new THREE.Line(g, m); if (dashed) l.computeLineDistances(); this.query.add(l); return l;
    };
    for (const s of sentences) {
      const col = new THREE.Color(s.color);
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.075, 24, 16), new THREE.MeshBasicMaterial({ color: col }));
      sphere.position.set(...s.pos); this.query.add(sphere);
      const light = getComputedStyle(document.documentElement).colorScheme === 'light';   // additive glow is invisible on a light background
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.glow, color: col, transparent: true, opacity: light ? 0.4 : 0.75, depthWrite: false, blending: light ? THREE.NormalBlending : THREE.AdditiveBlending }));
      halo.scale.setScalar(0.85); halo.position.set(...s.pos); this.query.add(halo);
      this._label(s.key, s.pos, 'sent sent-' + s.key);
      for (const p of s.links || []) line([s.pos, p], col, 0.45);
      if (showTraj && s.traj) {
        line(s.traj, col, 0.55);
        s.traj.forEach((p, i) => {
          if (i === layer) return;
          const dot = new THREE.Mesh(new THREE.SphereGeometry(0.018, 8, 6), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.6 }));
          dot.position.set(...p); this.query.add(dot);
        });
        this._label('layer 0', s.traj[0], 'tiny');
      }
      if (showWords) for (const w of s.words || []) {
        const r = 0.022 + 0.03 * Math.min(1, Math.max(0, w.weight ?? 0.5));
        const m = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 8), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.9 }));
        m.position.set(...w.pos); this.query.add(m);
        line([s.pos, w.pos], col, 0.16);
        this._label(w.text, w.pos, 'word word-' + s.key);
      }
    }
    if (sentences.length === 2) line([sentences[0].pos, sentences[1].pos], getComputedStyle(document.documentElement).getPropertyValue('--text').trim() || '#f0efec', 0.85, true);
  }

  /** Smoothly frame a set of points. */
  focus(points, pad = 3.3) {
    const box = new THREE.Box3().setFromPoints(points.map(p => new THREE.Vector3(...p)));
    const c = box.getCenter(new THREE.Vector3()), r = Math.max(0.9, box.getSize(new THREE.Vector3()).length() / 2);
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this.camAnim = { t0: performance.now(), ms: 1100, fromT: this.controls.target.clone(), toT: c,
      fromP: this.camera.position.clone(), toP: c.clone().add(dir.multiplyScalar(r * pad + 2.6)) };
  }
  resetView() { this.camAnim = { t0: performance.now(), ms: 900, fromT: this.controls.target.clone(), toT: new THREE.Vector3(), fromP: this.camera.position.clone(), toP: new THREE.Vector3(5.2, 3.2, 7.4) }; }

  frame(now) {
    const ease = t => t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
    if (this.anim) {
      const t = Math.min(1, (now - this.anim.t0) / this.anim.ms), e = ease(t), a = this.points.geometry.attributes.position;
      for (let i = 0; i < a.array.length; i++) a.array[i] = this.anim.from[i] + (this.anim.to[i] - this.anim.from[i]) * e;
      a.needsUpdate = true; if (t >= 1) this.anim = null;
    }
    if (this.camAnim) {
      const c = this.camAnim, t = Math.min(1, (now - c.t0) / c.ms), e = ease(t);
      this.controls.target.lerpVectors(c.fromT, c.toT, e); this.camera.position.lerpVectors(c.fromP, c.toP, e);
      if (t >= 1) this.camAnim = null;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    const w = this.el.clientWidth, h = this.el.clientHeight, v = new THREE.Vector3();
    for (const l of this.labels) {
      v.copy(l.pos).project(this.camera);
      const vis = v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05;
      l.el.style.display = vis ? '' : 'none';
      if (vis) l.el.style.transform = `translate(${((v.x + 1) / 2 * w).toFixed(1)}px, ${((1 - v.y) / 2 * h).toFixed(1)}px)`;
    }
  }

  _hit(ev) {
    if (!this.points) return -1;
    const r = this.renderer.domElement.getBoundingClientRect();
    const m = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(m, this.camera);
    const hits = this.ray.intersectObject(this.points).filter(h => !this.alphas || this.alphas[h.index] > 0.05);
    if (!hits.length) return -1;
    hits.sort((a, b) => a.distanceToRay - b.distanceToRay);
    return hits[0].index;
  }
  _bind() {
    const c = this.renderer.domElement; let down = null;
    c.addEventListener('pointermove', ev => { if (ev.buttons) return; this.onHover(this._hit(ev), ev); });
    c.addEventListener('pointerleave', ev => this.onHover(-1, ev));
    c.addEventListener('pointerdown', ev => { down = [ev.clientX, ev.clientY]; });
    c.addEventListener('pointerup', ev => {
      if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 4) return;
      const i = this._hit(ev); if (i >= 0) this.onPick(i, ev);
    });
  }
}
