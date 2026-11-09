(() => {
  'use strict';
  // Give native players an accessible name from their nearby task caption.
  document.querySelectorAll('figure').forEach((figure) => {
    const video = figure.querySelector('video');
    const caption = figure.querySelector('figcaption');
    if (video && caption) video.setAttribute('aria-label', caption.textContent.trim());
  });
})();
