/* Section nav highlighting and scroll reveal.
   Both are progressive: with JavaScript off, or with reduced motion requested,
   the page is fully readable and every figure is visible. */

(function () {
  "use strict";

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- reveal figures and blocks as they come into view --------------------
  var revealables = document.querySelectorAll(".reveal");
  if (reduced || !("IntersectionObserver" in window)) {
    revealables.forEach(function (el) { el.classList.add("is-visible"); });
  } else {
    var revealer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealer.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -12% 0px", threshold: 0.05 });
    revealables.forEach(function (el) { revealer.observe(el); });
  }

  // ---- mark the section the reader is in -----------------------------------
  var links = Array.prototype.slice.call(document.querySelectorAll(".section-nav a"));
  var sections = links
    .map(function (a) { return document.querySelector(a.getAttribute("href")); })
    .filter(Boolean);

  if (sections.length && "IntersectionObserver" in window) {
    var seen = new Map();
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { seen.set(e.target.id, e.intersectionRatio); });
      var best = null, bestRatio = 0;
      seen.forEach(function (ratio, id) {
        if (ratio > bestRatio) { bestRatio = ratio; best = id; }
      });
      links.forEach(function (a) {
        a.classList.toggle("active", bestRatio > 0 && a.getAttribute("href") === "#" + best);
      });
    }, { threshold: [0, 0.15, 0.35, 0.6, 0.9] });
    sections.forEach(function (s) { spy.observe(s); });
  }

  // ---- back to top ---------------------------------------------------------
  // Watching the intro leave the viewport avoids a scroll handler; the anchor
  // itself does the scrolling, so this only decides whether it is on screen.
  var totop = document.querySelector(".totop");
  var intro = document.querySelector(".intro");
  if (totop && intro && "IntersectionObserver" in window) {
    totop.classList.add("is-off");
    new IntersectionObserver(function (entries) {
      totop.classList.toggle("is-off", entries[0].isIntersecting);
    }, { threshold: 0 }).observe(intro);
  }

  // ---- keep Plotly charts sized to the column ------------------------------
  // Charts are configured responsive, but Plotly only measures on window
  // resize; a font swap or a late reveal can leave one at its initial width.
  function resizeCharts() {
    if (!window.Plotly) return;
    document.querySelectorAll("figure .chart > div").forEach(function (d) {
      try { window.Plotly.Plots.resize(d); } catch (err) { /* not yet drawn */ }
    });
  }
  window.addEventListener("load", resizeCharts);
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(resizeCharts);
  }
})();
