(function () {
  function setupSimulatorPlayback() {
    document.querySelectorAll('.simulator-grid video').forEach(function (video) {
      if (video.dataset.playbackReady) return;
      video.dataset.playbackReady = 'true';
      function useDoubleSpeed() {
        video.defaultPlaybackRate = 2;
        video.playbackRate = 2;
      }
      useDoubleSpeed();
      video.addEventListener('loadedmetadata', useDoubleSpeed, { once: true });
    });
  }

  function setup() {
    setupSimulatorPlayback();
    const button = document.getElementById('copy-bibtex');
    const code = document.getElementById('bibtex');
    const status = document.getElementById('copy-status');
    if (!button || !code || button.dataset.ready) return;
    button.dataset.ready = 'true';
    button.addEventListener('click', async function () {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = 'Copied!';
        status.textContent = 'BibTeX copied to clipboard.';
        window.setTimeout(() => { button.textContent = 'Copy BibTeX'; }, 2000);
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(code);
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = 'Select and copy the highlighted BibTeX.';
      }
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', setup);
  else setup();
})();
