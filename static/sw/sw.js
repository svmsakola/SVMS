self.addEventListener('install', function (e) {
  console.log('[ServiceWorker] Install');
  self.skipWaiting();
});

self.addEventListener('fetch', function (event) {
  event.respondWith(fetch(event.request));
});
