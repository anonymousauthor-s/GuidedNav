/* GuidedNav's decorative, articulated X2-inspired walking mascot. */
(() => {
  'use strict';

  const WIDTH = 160;
  const HEIGHT = 188;
  const TAU = Math.PI * 2;
  const PERIOD = 1700;

  function roundedPath(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function ellipse(ctx, x, y, rx, ry, fill, stroke) {
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, TAU);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
  }

  function shell(ctx, x1, y1, x2, y2, radius, far = false) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const length = Math.hypot(dx, dy);
    ctx.save();
    ctx.translate(x1, y1);
    ctx.rotate(Math.atan2(dy, dx) - Math.PI / 2);
    const gradient = ctx.createLinearGradient(-radius, 0, radius, 0);
    gradient.addColorStop(0, far ? '#b6c1c7' : '#d2dce0');
    gradient.addColorStop(0.32, far ? '#e5eaed' : '#ffffff');
    gradient.addColorStop(0.7, far ? '#d5dde1' : '#f7f9fa');
    gradient.addColorStop(1, far ? '#95a5af' : '#b1bfc7');
    roundedPath(ctx, -radius, 0, radius * 2, length, radius * 0.8);
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.lineWidth = 0.65;
    ctx.strokeStyle = far ? '#8c9aa2' : '#acb8bf';
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-radius * 0.45, radius * 0.7);
    ctx.lineTo(-radius * 0.45, length - radius * 0.7);
    ctx.lineWidth = 1;
    ctx.strokeStyle = far ? 'rgba(255,255,255,.36)' : 'rgba(255,255,255,.9)';
    ctx.stroke();
    ctx.restore();
  }

  function joint(ctx, x, y, radius, far = false) {
    const gradient = ctx.createRadialGradient(x - radius * 0.35, y - radius * 0.45, 0.3, x, y, radius);
    gradient.addColorStop(0, far ? '#6b7780' : '#636c75');
    gradient.addColorStop(0.5, '#30363d');
    gradient.addColorStop(1, '#171d24');
    ellipse(ctx, x, y, radius, radius, gradient, '#1b252c');
  }

  // The foot moves backward while planted, then lifts and returns forward.
  function step(phase, ground) {
    const stance = 0.56;
    if (phase < stance) {
      return { x: 13 - 26 * phase / stance, y: ground, tilt: 0 };
    }
    const progress = (phase - stance) / (1 - stance);
    const smooth = progress * progress * (3 - 2 * progress);
    return {
      x: -13 + 26 * smooth,
      y: ground - 12 * Math.sin(Math.PI * progress),
      tilt: -0.22 * Math.sin(Math.PI * progress)
    };
  }

  // Two-link inverse kinematics: the knee bends forward as the foot lifts.
  function knee(hip, ankle, thigh = 24, shin = 24) {
    const dx = ankle.x - hip.x;
    const dy = ankle.y - hip.y;
    const distance = Math.min(Math.hypot(dx, dy), thigh + shin - 0.01);
    const angle = Math.atan2(dy, dx);
    const offset = Math.acos(Math.max(-1, Math.min(1,
      (thigh * thigh + distance * distance - shin * shin) / (2 * thigh * distance)
    )));
    return { x: hip.x + thigh * Math.cos(angle - offset), y: hip.y + thigh * Math.sin(angle - offset) };
  }

  function shoe(ctx, foot, far) {
    ctx.save();
    ctx.translate(foot.x, foot.y);
    ctx.rotate(foot.tilt);
    const gold = ctx.createLinearGradient(-7, -4, 15, 8);
    gold.addColorStop(0, far ? '#e3c34b' : '#fff48a');
    gold.addColorStop(0.34, far ? '#efc336' : '#ffdb32');
    gold.addColorStop(1, far ? '#d39c12' : '#f2b313');
    ctx.beginPath();
    ctx.moveTo(-7, 0);
    ctx.quadraticCurveTo(-5, -6, 2, -6);
    ctx.quadraticCurveTo(14, -6, 19, 4);
    ctx.lineTo(19, 7);
    ctx.quadraticCurveTo(6, 11, -8, 7);
    ctx.closePath();
    ctx.fillStyle = gold;
    ctx.fill();
    ctx.strokeStyle = '#bf9624';
    ctx.lineWidth = 0.65;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-8, 5);
    ctx.quadraticCurveTo(4, 9, 19, 5);
    ctx.lineTo(19, 9);
    ctx.quadraticCurveTo(5, 13, -8, 9);
    ctx.closePath();
    ctx.fillStyle = far ? '#303841' : '#252c34';
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(0, -3.3);
    ctx.quadraticCurveTo(8, -3.4, 12, 0);
    ctx.strokeStyle = far ? 'rgba(255,249,177,.5)' : 'rgba(255,255,220,.9)';
    ctx.lineWidth = 1.35;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.restore();
  }

  function leg(ctx, hip, foot, far) {
    const bend = knee(hip, foot);
    joint(ctx, hip.x, hip.y, 7, far);
    const thighStart = { x: hip.x + (bend.x - hip.x) * 0.13, y: hip.y + (bend.y - hip.y) * 0.13 };
    const thighEnd = { x: hip.x + (bend.x - hip.x) * 0.79, y: hip.y + (bend.y - hip.y) * 0.79 };
    shell(ctx, thighStart.x, thighStart.y, thighEnd.x, thighEnd.y, far ? 6 : 7, far);
    joint(ctx, bend.x, bend.y, 6, far);
    const shinStart = { x: bend.x + (foot.x - bend.x) * 0.22, y: bend.y + (foot.y - bend.y) * 0.22 };
    const shinEnd = { x: bend.x + (foot.x - bend.x) * 0.89, y: bend.y + (foot.y - bend.y) * 0.89 };
    shell(ctx, shinStart.x, shinStart.y, shinEnd.x, shinEnd.y, far ? 6 : 6.5, far);
    joint(ctx, foot.x, foot.y, 4, far);
    shoe(ctx, foot, far);
  }

  function arm(ctx, shoulder, phase, far) {
    const angle = -0.43 * Math.cos(phase * TAU);
    const forearmAngle = angle + 0.32;
    const elbow = { x: shoulder.x + 19 * Math.sin(angle), y: shoulder.y + 19 * Math.cos(angle) };
    const hand = { x: elbow.x + 17 * Math.sin(forearmAngle), y: elbow.y + 17 * Math.cos(forearmAngle) };
    joint(ctx, shoulder.x, shoulder.y, far ? 6 : 7, far);
    shell(ctx,
      shoulder.x + (elbow.x - shoulder.x) * 0.2,
      shoulder.y + (elbow.y - shoulder.y) * 0.2,
      shoulder.x + (elbow.x - shoulder.x) * 0.82,
      shoulder.y + (elbow.y - shoulder.y) * 0.82,
      far ? 5 : 6, far);
    joint(ctx, elbow.x, elbow.y, 5, far);
    shell(ctx,
      elbow.x + (hand.x - elbow.x) * 0.15,
      elbow.y + (hand.y - elbow.y) * 0.15,
      elbow.x + (hand.x - elbow.x) * 0.78,
      elbow.y + (hand.y - elbow.y) * 0.78,
      far ? 4.5 : 5.5, far);
    joint(ctx, hand.x, hand.y, far ? 4.5 : 5, far);
    ctx.beginPath();
    ctx.moveTo(hand.x + 1, hand.y - 3);
    ctx.lineTo(hand.x + 1.5, hand.y + 1.8);
    ctx.strokeStyle = 'rgba(137,150,159,.35)';
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }

  function torso(ctx, bob) {
    const gradient = ctx.createLinearGradient(61, 83 + bob, 100, 119 + bob);
    gradient.addColorStop(0, '#65717b');
    gradient.addColorStop(0.32, '#37424c');
    gradient.addColorStop(1, '#1a252e');
    ctx.beginPath();
    ctx.moveTo(69, 78 + bob);
    ctx.bezierCurveTo(81, 75 + bob, 96, 83 + bob, 100, 98 + bob);
    ctx.bezierCurveTo(102, 111 + bob, 98, 121 + bob, 86, 126 + bob);
    ctx.bezierCurveTo(69, 128 + bob, 59, 118 + bob, 59, 104 + bob);
    ctx.bezierCurveTo(59, 91 + bob, 60, 82 + bob, 69, 78 + bob);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.strokeStyle = '#18212a';
    ctx.lineWidth = 0.85;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(67, 83 + bob);
    ctx.bezierCurveTo(61, 91 + bob, 61, 104 + bob, 64, 113 + bob);
    ctx.strokeStyle = '#dce4e8';
    ctx.lineWidth = 3.5;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(65, 116 + bob);
    ctx.quadraticCurveTo(82, 126 + bob, 97, 115 + bob);
    ctx.strokeStyle = '#6e7c86';
    ctx.lineWidth = 1.1;
    ctx.stroke();
    ellipse(ctx, 93, 91 + bob, 1.5, 2.3, '#7eb1ff');
  }

  function head(ctx, bob, phase) {
    ctx.save();
    ctx.translate(81, 46 + bob);
    ctx.rotate(-0.025 + Math.sin(phase * TAU) * 0.012);
    ctx.translate(-81, -46);
    ellipse(ctx, 80, 76, 10, 5, '#28323a', '#121b22');
    const white = ctx.createLinearGradient(44, 13, 117, 73);
    white.addColorStop(0, '#d7e0e5');
    white.addColorStop(0.22, '#ffffff');
    white.addColorStop(0.58, '#f9fafb');
    white.addColorStop(1, '#b7c4cd');
    ctx.beginPath();
    ctx.moveTo(58, 13);
    ctx.bezierCurveTo(84, 5, 113, 14, 121, 33);
    ctx.bezierCurveTo(128, 51, 117, 71, 96, 77);
    ctx.bezierCurveTo(74, 84, 49, 70, 44, 51);
    ctx.bezierCurveTo(39, 32, 44, 19, 58, 13);
    ctx.closePath();
    ctx.fillStyle = white;
    ctx.fill();
    ctx.strokeStyle = '#adbcc6';
    ctx.lineWidth = 0.8;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(60, 18);
    ctx.bezierCurveTo(81, 10, 106, 20, 114, 30);
    ctx.strokeStyle = 'rgba(255,255,255,.92)';
    ctx.lineWidth = 2.1;
    ctx.stroke();

    const face = ctx.createLinearGradient(74, 23, 110, 67);
    face.addColorStop(0, '#34414b');
    face.addColorStop(0.26, '#18212b');
    face.addColorStop(1, '#0d1720');
    ctx.beginPath();
    ctx.moveTo(80, 25);
    ctx.bezierCurveTo(96, 22, 115, 28, 118, 37);
    ctx.bezierCurveTo(120, 47, 114, 62, 105, 66);
    ctx.bezierCurveTo(98, 70, 81, 69, 76, 64);
    ctx.bezierCurveTo(69, 58, 65, 41, 71, 32);
    ctx.quadraticCurveTo(74, 27, 80, 25);
    ctx.closePath();
    ctx.fillStyle = face;
    ctx.fill();
    ctx.strokeStyle = '#88959f';
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(80, 29);
    ctx.quadraticCurveTo(99, 26, 111, 34);
    ctx.strokeStyle = 'rgba(225,240,248,.15)';
    ctx.lineWidth = 1.3;
    ctx.stroke();

    ctx.save();
    ctx.shadowColor = '#64a8ff';
    ctx.shadowBlur = 5;
    ctx.strokeStyle = '#80c1ff';
    ctx.lineWidth = 3.4;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(79, 49);
    ctx.quadraticCurveTo(83, 39, 89, 48);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(101, 49);
    ctx.quadraticCurveTo(105, 42, 109, 50);
    ctx.stroke();
    ctx.restore();

    const ear = ctx.createLinearGradient(41, 32, 57, 58);
    ear.addColorStop(0, '#ffffff');
    ear.addColorStop(1, '#bdcbd3');
    ellipse(ctx, 49, 46, 10, 16, ear, '#afbec7');
    ellipse(ctx, 48, 47, 7, 12, '#242e37', '#6c7b88');
    ctx.beginPath();
    ctx.ellipse(48, 47, 4.6, 8.7, -0.08, 0, TAU);
    ctx.strokeStyle = '#5997ef';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ellipse(ctx, 47.5, 46.5, 2.6, 6.4, '#172732');
    ctx.restore();
  }

  function paint(ctx, phase, scaleX, scaleY, reduced) {
    ctx.setTransform(scaleX, 0, 0, scaleY, 0, 0);
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    const bob = reduced ? 0 : -1.5 + 1.1 * Math.cos(phase * TAU * 2);
    const shadow = ctx.createRadialGradient(82, 177, 2, 82, 177, 39);
    shadow.addColorStop(0, 'rgba(31,62,112,.15)');
    shadow.addColorStop(1, 'rgba(31,62,112,0)');
    ellipse(ctx, 82, 177, 39, 5.5, shadow);
    const farPhase = (phase + 0.5) % 1;
    const farStep = step(farPhase, 163);
    const nearStep = step(phase, 166);
    const farFoot = { x: 85 + farStep.x, y: farStep.y, tilt: farStep.tilt };
    const nearFoot = { x: 75 + nearStep.x, y: nearStep.y, tilt: nearStep.tilt };
    leg(ctx, { x: 85, y: 121 + bob }, farFoot, true);
    arm(ctx, { x: 87, y: 85 + bob }, farPhase, true);
    leg(ctx, { x: 75, y: 122 + bob }, nearFoot, false);
    torso(ctx, bob);
    arm(ctx, { x: 64, y: 85 + bob }, phase, false);
    head(ctx, bob, phase);
  }

  document.querySelectorAll('canvas[data-walking-mascot]').forEach((canvas) => {
    if (canvas.dataset.ready === 'true') return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let intersecting = true;
    let request = 0;
    let elapsed = 0;
    let lastTime = 0;
    let sx = 1;
    let sy = 1;

    const render = () => paint(ctx, reducedMotion.matches ? 0 : (elapsed / PERIOD) % 1, sx, sy, reducedMotion.matches);
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 3);
      canvas.width = Math.max(1, Math.round((rect.width || WIDTH) * ratio));
      canvas.height = Math.max(1, Math.round((rect.height || HEIGHT) * ratio));
      sx = canvas.width / WIDTH;
      sy = canvas.height / HEIGHT;
      render();
    };
    const visible = () => !document.hidden && intersecting && !reducedMotion.matches;
    const frame = (time) => {
      request = 0;
      if (!visible()) { lastTime = 0; return; }
      if (lastTime) elapsed += Math.min(time - lastTime, 80);
      lastTime = time;
      render();
      request = window.requestAnimationFrame(frame);
    };
    const sync = () => {
      if (visible()) {
        if (!request) { lastTime = 0; request = window.requestAnimationFrame(frame); }
      } else {
        if (request) window.cancelAnimationFrame(request);
        request = 0;
        lastTime = 0;
        if (reducedMotion.matches) render();
      }
    };

    resize();
    canvas.dataset.ready = 'true';
    document.addEventListener('visibilitychange', sync);
    if (reducedMotion.addEventListener) reducedMotion.addEventListener('change', sync);
    else reducedMotion.addListener(sync);
    if ('ResizeObserver' in window) new ResizeObserver(resize).observe(canvas);
    else window.addEventListener('resize', resize, { passive: true });
    if ('IntersectionObserver' in window) {
      new IntersectionObserver((entries) => {
        intersecting = entries.some((entry) => entry.isIntersecting);
        sync();
      }, { rootMargin: '24px' }).observe(canvas);
    }
    sync();
  });
})();
