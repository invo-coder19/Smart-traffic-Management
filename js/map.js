// ============================================================
// Smart Traffic Crime Detection — Map Module (Leaflet.js)
// ============================================================

const TrafficMap = (() => {
  let map = null;
  let markers = [];
  let trafficLayer = null;

  const cameraIcon = (status) => L.divIcon({
    className: '',
    html: `<div class="map-camera-marker ${status === 'Online' ? 'online' : 'offline'}">
             <i class="fas fa-video"></i>
             <div class="marker-pulse"></div>
           </div>`,
    iconSize: [36, 36],
    iconAnchor: [18, 18],
  });

  const hotspotIcon = () => L.divIcon({
    className: '',
    html: `<div class="map-hotspot-marker"><i class="fas fa-fire"></i></div>`,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
  });

  const init = (containerId) => {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (map) { map.remove(); map = null; }

    map = L.map(containerId, {
      center: [28.6139, 77.2090],
      zoom: 12,
      zoomControl: false,
      attributionControl: false,
    });

    // Dark tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '©OpenStreetMap ©CARTO',
      subdomains: 'abcd',
      maxZoom: 20,
    }).addTo(map);

    L.control.zoom({ position: 'topright' }).addTo(map);

    // Add camera markers
    const cameras = Array.from({ length: 16 }, (_, i) => TrafficData.generateCamera(i));
    cameras.forEach(cam => {
      const marker = L.marker([cam.lat, cam.lng], { icon: cameraIcon(cam.status) })
        .addTo(map)
        .bindPopup(`
          <div class="map-popup">
            <div class="popup-header">
              <i class="fas fa-video"></i>
              <strong>${cam.id}</strong>
              <span class="popup-status ${cam.status === 'Online' ? 'online' : 'offline'}">${cam.status}</span>
            </div>
            <div class="popup-body">
              <div class="popup-row"><span>Location:</span><span>${cam.location}</span></div>
              <div class="popup-row"><span>Resolution:</span><span>${cam.resolution}</span></div>
              <div class="popup-row"><span>FPS:</span><span>${cam.fps}</span></div>
              <div class="popup-row"><span>AI Model:</span><span>${cam.aiModel}</span></div>
              <div class="popup-row"><span>Detections:</span><span>${cam.detections}</span></div>
            </div>
          </div>
        `, {
          className: 'custom-popup',
          maxWidth: 250,
        });
      markers.push(marker);
    });

    // Add accident hotspots
    const hotspots = [
      [28.632, 77.219], [28.598, 77.243], [28.651, 77.198],
      [28.614, 77.185], [28.627, 77.231],
    ];
    hotspots.forEach(([lat, lng]) => {
      L.marker([lat, lng], { icon: hotspotIcon() })
        .addTo(map)
        .bindPopup('<div class="map-popup"><strong style="color:#ff3366">⚠ Accident Hotspot</strong><br>High incident frequency zone</div>', { className: 'custom-popup' });
    });

    // Traffic density circles
    const densityPoints = [
      { lat: 28.6139, lng: 77.2090, intensity: 0.9 },
      { lat: 28.632, lng: 77.219, intensity: 0.7 },
      { lat: 28.598, lng: 77.243, intensity: 0.5 },
      { lat: 28.651, lng: 77.198, intensity: 0.8 },
      { lat: 28.621, lng: 77.175, intensity: 0.4 },
    ];

    densityPoints.forEach(p => {
      const color = p.intensity > 0.7 ? '#ff3366' : p.intensity > 0.4 ? '#ff6b35' : '#ffd700';
      L.circle([p.lat, p.lng], {
        radius: 300 * p.intensity + 100,
        fillColor: color,
        fillOpacity: 0.15,
        color: color,
        opacity: 0.3,
        weight: 1,
      }).addTo(map);
    });

    // Live alert animation every 10s
    setInterval(() => {
      const cam = cameras[Math.floor(Math.random() * cameras.length)];
      const alertCircle = L.circle([cam.lat, cam.lng], {
        radius: 150,
        fillColor: '#ff3366',
        fillOpacity: 0.4,
        color: '#ff3366',
        opacity: 0.8,
        weight: 2,
      }).addTo(map);
      setTimeout(() => map && map.removeLayer(alertCircle), 3000);
    }, 10000);
  };

  const invalidateSize = () => {
    if (map) setTimeout(() => map.invalidateSize(), 100);
  };

  return { init, invalidateSize };
})();

window.TrafficMap = TrafficMap;
