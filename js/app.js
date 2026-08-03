// ============================================================
// Smart Traffic Crime Detection — App Controller
// ============================================================

const App = (() => {
  let currentPage = 'dashboard';
  let violationsData = [];
  let vehiclesData = [];
  let challansData = [];
  let camerasData = [];
  let usersData = [];
  let activityInterval = null;
  let cameraAnimInterval = null;
  let statsCounterDone = false;
  let lightMode = false;

  // ── Page definitions ─────────────────────────────────────────
  const pages = [
    { id: 'dashboard',   label: 'Dashboard',         icon: 'fa-th-large',       group: 'MAIN' },
    { id: 'live-feed',   label: 'Live Camera Feed',  icon: 'fa-video',          group: 'MAIN', badge: 'LIVE' },
    { id: 'ai-detection',label: 'AI Detection',      icon: 'fa-robot',          group: 'MAIN' },
    { id: 'violations',  label: 'Violations',        icon: 'fa-exclamation-triangle', group: 'ENFORCEMENT' },
    { id: 'vehicles',    label: 'Vehicles',          icon: 'fa-car',            group: 'ENFORCEMENT' },
    { id: 'challans',    label: 'Challans',          icon: 'fa-file-invoice',   group: 'ENFORCEMENT' },
    { id: 'analytics',   label: 'Analytics',         icon: 'fa-chart-line',     group: 'INSIGHTS' },
    { id: 'reports',     label: 'Reports',           icon: 'fa-file-alt',       group: 'INSIGHTS' },
    { id: 'cameras',     label: 'Camera Management', icon: 'fa-camera',         group: 'MANAGEMENT' },
    { id: 'users',       label: 'Users',             icon: 'fa-users',          group: 'MANAGEMENT' },
    { id: 'settings',    label: 'Settings',          icon: 'fa-cog',            group: 'MANAGEMENT' },
  ];

  // ── Generate data ─────────────────────────────────────────────
  const initData = () => {
    violationsData = Array.from({ length: 60 }, (_, i) => TrafficData.generateViolation(`VIO-${String(i+1).padStart(6,'0')}`));
    vehiclesData   = Array.from({ length: 40 }, () => ({
      plate: TrafficData.numberPlates(),
      type: ['Car','Motorcycle','Truck','Bus','Auto','Van','SUV'][Math.floor(Math.random()*7)],
      violations: Math.floor(Math.random() * 6),
      challans: Math.floor(Math.random() * 4),
      lastSeen: TrafficData.randomTime(),
    }));
    challansData   = violationsData.slice(0, 40).map(v => ({
      ...v, challanId: `CH-${String(Math.floor(Math.random()*900000+100000))}`,
      issueDate: '2026-08-03', dueDate: '2026-08-17',
    }));
    camerasData    = Array.from({ length: 16 }, (_, i) => TrafficData.generateCamera(i));
    usersData      = Array.from({ length: 12 }, (_, i) => TrafficData.generateUser(i));
  };

  // ── Sidebar render ───────────────────────────────────────────
  const renderSidebar = () => {
    const nav = document.getElementById('sidebar-nav');
    let currentGroup = '';
    let html = '';
    pages.forEach(p => {
      if (p.group !== currentGroup) {
        currentGroup = p.group;
        html += `<div class="nav-group-label">${p.group}</div>`;
      }
      const badge = p.badge ? `<span class="nav-badge">${p.badge}</span>` : '';
      html += `
        <a class="nav-item ${p.id === currentPage ? 'active' : ''}" data-page="${p.id}" onclick="App.navigate('${p.id}')">
          <span class="nav-icon"><i class="fas ${p.icon}"></i></span>
          <span class="nav-label">${p.label}</span>
          ${badge}
        </a>`;
    });
    nav.innerHTML = html;
  };

  // ── Navigate ──────────────────────────────────────────────────
  const navigate = (pageId) => {
    currentPage = pageId;
    renderSidebar();
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    const section = document.getElementById(`page-${pageId}`);
    if (section) {
      section.classList.add('active');
      renderPage(pageId);
    }
    // Update breadcrumb
    const pg = pages.find(p => p.id === pageId);
    const titleEl = document.getElementById('page-title');
    const subEl = document.getElementById('page-sub');
    if (titleEl && pg) titleEl.textContent = pg.label;
    if (subEl) subEl.textContent = 'Smart Traffic Crime Detection System';
    // Close mobile sidebar
    document.getElementById('sidebar')?.classList.remove('mobile-open');
    document.getElementById('sidebar-overlay')?.classList.remove('show');
    TrafficMap.invalidateSize();
  };

  // ── Render specific pages ─────────────────────────────────────
  const renderPage = (pageId) => {
    switch (pageId) {
      case 'dashboard':   renderDashboard(); break;
      case 'live-feed':   renderLiveFeed(); break;
      case 'ai-detection':renderAIDetection(); break;
      case 'violations':  renderViolations(); break;
      case 'vehicles':    renderVehicles(); break;
      case 'challans':    renderChallans(); break;
      case 'analytics':   renderAnalytics(); break;
      case 'reports':     renderReports(); break;
      case 'cameras':     renderCameras(); break;
      case 'users':       renderUsers(); break;
      case 'settings':    renderSettings(); break;
    }
  };

  // ══════════════════════════════════════════════════════════════
  //  DASHBOARD
  // ══════════════════════════════════════════════════════════════
  const renderDashboard = () => {
    const el = document.getElementById('page-dashboard');
    const s = TrafficData.stats;

    el.innerHTML = `
      <!-- Stats Grid -->
      <div class="stats-grid mb-6">
        ${[
          { label:'Active Cameras',          value: s.activeCameras,                  icon:'fa-video',             color1:'#00d4ff', color2:'#0099cc', trend:'+3 today',    up:true },
          { label:'Live Vehicles',           value: s.liveVehicles.toLocaleString(),   icon:'fa-car',               color1:'#00ff88', color2:'#00cc6a', trend:'+12% vs avg', up:true },
          { label:'Total Violations Today',  value: s.totalViolations.toLocaleString(),icon:'fa-exclamation-circle',color1:'#ff3366', color2:'#cc0033', trend:'-8% vs yesterday', up:false },
          { label:'Number Plate Detections', value: s.numberPlateDetections.toLocaleString(), icon:'fa-id-card',  color1:'#9b59b6', color2:'#7c3aed', trend:'+5%',          up:true },
          { label:'Pending Challans',        value: s.pendingChallans,                 icon:'fa-file-invoice',      color1:'#ff6b35', color2:'#cc4411', trend: '67 new today',up:false },
          { label:'Revenue Collected',       value:'₹'+formatNum(s.revenueCollected),  icon:'fa-rupee-sign',        color1:'#ffd700', color2:'#ccaa00', trend:'+22% this month', up:true },
          { label:'AI Detection Accuracy',   value: s.aiAccuracy+'%',                  icon:'fa-robot',             color1:'#00ff88', color2:'#00cc6a', trend:'↑ YOLOv9',    up:true },
          { label:'System Health',           value: s.systemHealth+'%',                icon:'fa-server',            color1:'#00d4ff', color2:'#0099cc', trend:'All nominal', up:true },
        ].map(c => `
          <div class="stat-card" style="--accent-start:${c.color1};--accent-end:${c.color2}">
            <i class="fas ${c.icon} stat-bg-icon"></i>
            <div class="stat-icon-wrap" style="background:linear-gradient(135deg,${c.color1}22,${c.color2}11);color:${c.color1}">
              <i class="fas ${c.icon}"></i>
            </div>
            <div class="stat-label">${c.label}</div>
            <div class="stat-value counter" data-target="${c.value}">${c.value}</div>
            <div class="stat-trend ${c.up ? 'trend-up' : 'trend-down'}">
              <i class="fas ${c.up ? 'fa-arrow-up' : 'fa-arrow-down'}"></i>
              <span>${c.trend}</span>
            </div>
          </div>
        `).join('')}
      </div>

      <!-- Live Camera Preview + Recent Violations -->
      <div class="charts-grid mb-6">
        <div class="chart-col-8">
          <div class="card">
            <div class="card-header">
              <span class="card-title-main">Live Camera Feed</span>
              <div class="flex gap-2">
                <span class="ai-live-badge flex items-center gap-2"><span class="ai-live-dot"></span>LIVE</span>
                <button class="btn btn-ghost btn-sm" onclick="App.navigate('live-feed')"><i class="fas fa-expand-alt"></i> Full View</button>
              </div>
            </div>
            <div class="camera-grid" id="dash-camera-grid" style="grid-template-columns: repeat(4,1fr)">
              ${renderCameraCards(8)}
            </div>
          </div>
        </div>
        <div class="chart-col-4">
          <div class="card" style="height: 100%">
            <div class="card-header">
              <span class="card-title-main">Violation Breakdown</span>
              <span class="text-muted text-xs">Today</span>
            </div>
            <div class="chart-container">
              <canvas id="vio-donut-chart"></canvas>
            </div>
          </div>
        </div>
      </div>

      <!-- Charts Row -->
      <div class="charts-grid mb-6">
        <div class="chart-col-8">
          <div class="card">
            <div class="card-header">
              <span class="card-title-main">Hourly Traffic Flow</span>
              <div class="flex gap-2">
                <button class="btn btn-ghost btn-sm active" id="btn-hour">24H</button>
                <button class="btn btn-ghost btn-sm" id="btn-week">7D</button>
              </div>
            </div>
            <div class="chart-container">
              <canvas id="hourly-chart"></canvas>
            </div>
          </div>
        </div>
        <div class="chart-col-4">
          <div class="card" style="height:100%">
            <div class="card-header">
              <span class="card-title-main">AI Accuracy</span>
              <span class="glow-text-green font-bold">97.3%</span>
            </div>
            <div class="chart-container chart-container-sm" style="height:140px">
              <canvas id="accuracy-gauge"></canvas>
            </div>
            <div style="text-align:center;margin-top:8px">
              <div class="stat-value glow-text-green" style="font-size:36px">97.3<span style="font-size:16px">%</span></div>
              <div class="text-muted text-sm">YOLOv9 Model Active</div>
            </div>
            <div class="divider"></div>
            ${['Helmet', 'Vehicles', 'Plates', 'Speed'].map((l,i) => `
              <div class="progress-bar-wrap">
                <div class="progress-bar-label"><span>${l} Detection</span><span style="color:var(--neon-green)">${(93+i*1.5).toFixed(1)}%</span></div>
                <div class="progress-bar"><div class="progress-fill" style="width:${93+i*1.5}%;background:linear-gradient(90deg,var(--neon-blue),var(--neon-green))"></div></div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>

      <!-- Map + Activity -->
      <div class="charts-grid mb-6">
        <div class="chart-col-8">
          <div class="card" style="padding:0;overflow:hidden">
            <div class="card-header" style="padding:16px 20px">
              <span class="card-title-main">Live City Map</span>
              <div class="flex gap-2">
                <span class="badge badge-online">● 142 Cameras Online</span>
                <button class="btn btn-ghost btn-sm"><i class="fas fa-expand"></i></button>
              </div>
            </div>
            <div id="map-container"></div>
            <div class="map-legend">
              <div class="map-legend-title">Legend</div>
              <div class="map-legend-item"><div class="map-legend-dot" style="background:var(--neon-blue)"></div> Camera Online</div>
              <div class="map-legend-item"><div class="map-legend-dot" style="background:var(--neon-red)"></div> Camera Offline</div>
              <div class="map-legend-item"><div class="map-legend-dot" style="background:var(--neon-red);opacity:0.5;width:14px;height:14px;border-radius:4px"></div> Hotspot</div>
              <div class="map-legend-item"><div class="map-legend-dot" style="background:var(--neon-orange)"></div> High Density</div>
            </div>
          </div>
        </div>
        <div class="chart-col-4">
          <div class="card" style="height:100%;padding:0;overflow:hidden">
            <div class="card-header" style="padding:16px 20px">
              <span class="card-title-main">Live Activity Feed</span>
              <span class="ai-live-badge"><span class="ai-live-dot"></span>LIVE</span>
            </div>
            <div class="activity-feed" id="activity-feed">
              ${renderActivityItems(10)}
            </div>
          </div>
        </div>
      </div>

      <!-- Recent Violations Table -->
      <div class="card mb-6">
        <div class="card-header">
          <span class="card-title-main">Recent Violations</span>
          <div class="flex gap-2">
            <button class="btn btn-ghost btn-sm"><i class="fas fa-filter"></i> Filter</button>
            <button class="btn btn-primary btn-sm" onclick="App.navigate('violations')"><i class="fas fa-arrow-right"></i> View All</button>
          </div>
        </div>
        <div class="table-wrapper">
          ${renderViolationTable(violationsData.slice(0, 8))}
        </div>
      </div>

      <!-- AI Insights -->
      <div class="card mb-6">
        <div class="card-header">
          <span class="card-title-main"><i class="fas fa-brain" style="color:var(--neon-blue)"></i> AI Predictive Insights</span>
          <span class="text-muted text-xs">Updated 2 min ago</span>
        </div>
        <div class="insights-grid">
          ${[
            { icon:'🔴', title:'Peak Violation Hour', value:'08:00–09:00', trend:'Morning rush +34%', color:'#ff3366' },
            { icon:'⚡', title:'Highest Risk Zone',   value:'MG Road',    trend:'78 incidents this week', color:'#ff6b35' },
            { icon:'🎯', title:'Top Violation',       value:'Overspeeding', trend:'31.2% of all violations', color:'#ffd700' },
            { icon:'📈', title:'Violation Trend',     value:'+12.4%',     trend:'vs last week', color:'#00d4ff' },
            { icon:'🚗', title:'Busiest Camera',      value:'CAM-003',    trend:'4,821 detections today', color:'#00ff88' },
            { icon:'💰', title:'Revenue Target',      value:'87.3%',      trend:'₹28.4L of ₹32.5L goal', color:'#9b59b6' },
          ].map(i => `
            <div class="insight-card" style="--insight-color:${i.color}">
              <div class="insight-icon">${i.icon}</div>
              <div class="insight-title">${i.title}</div>
              <div class="insight-value" style="color:${i.color}">${i.value}</div>
              <div class="insight-trend">${i.trend}</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;

    // Init charts & map
    setTimeout(() => {
      Charts.initViolationDonut('vio-donut-chart');
      Charts.initHourlyTraffic('hourly-chart');
      Charts.initAccuracyGauge('accuracy-gauge', 97.3);
      TrafficMap.init('map-container');
      startCameraAnimations();
      startActivityFeed();
      animateCounters();
    }, 100);
  };

  // ── Camera Cards HTML ─────────────────────────────────────────
  const renderCameraCards = (count) => {
    const cams = camerasData.slice(0, count);
    return cams.map(cam => `
      <div class="camera-feed-card ${cam.status === 'Online' ? 'active-cam' : ''}" data-cam="${cam.id}">
        <canvas class="camera-feed-canvas cam-bg" id="feed-${cam.id}" width="320" height="180"></canvas>
        <div class="cam-overlay"></div>
        <div class="cam-top-info">
          <span class="cam-timestamp" id="ts-${cam.id}">--:--:--</span>
          ${cam.status === 'Online' ? '<span class="cam-live-badge">● LIVE</span>' : '<span class="cam-live-badge" style="background:var(--neon-orange)">OFFLINE</span>'}
        </div>
        <div class="cam-info">
          <span class="cam-id">${cam.id} | ${cam.location.split(' ')[0]}</span>
          <div class="cam-status-dot ${cam.status === 'Online' ? '' : 'offline'}"></div>
        </div>
      </div>
    `).join('');
  };

  // ── Activity Items ────────────────────────────────────────────
  const renderActivityItems = (count) => {
    return Array.from({ length: count }, () => {
      const v = violationsData[Math.floor(Math.random() * violationsData.length)];
      const icons = [
        { icon: 'fa-exclamation-circle', bg: 'rgba(255,51,102,0.15)', color: '#ff3366' },
        { icon: 'fa-video', bg: 'rgba(0,212,255,0.15)', color: '#00d4ff' },
        { icon: 'fa-car', bg: 'rgba(0,255,136,0.15)', color: '#00ff88' },
        { icon: 'fa-file-invoice', bg: 'rgba(255,107,53,0.15)', color: '#ff6b35' },
      ];
      const ic = icons[Math.floor(Math.random() * icons.length)];
      return `
        <div class="activity-item">
          <div class="activity-icon" style="background:${ic.bg};color:${ic.color}">
            <i class="fas ${ic.icon}"></i>
          </div>
          <div class="activity-text">
            <div class="activity-main">${v.violation} — <span class="table-plate">${v.vehicleNumber}</span></div>
            <div class="activity-sub">${v.location} · ${v.camera}</div>
          </div>
          <div class="activity-time">${v.time}</div>
        </div>
      `;
    }).join('');
  };

  // ── Violation Table ───────────────────────────────────────────
  const renderViolationTable = (data) => `
    <table class="data-table">
      <thead>
        <tr>
          <th>Vehicle No.</th><th>Violation</th><th>Camera</th>
          <th>Time</th><th>Fine</th><th>Status</th><th>Officer</th><th>Actions</th>
        </tr>
      </thead>
      <tbody>
        ${data.map(v => `
          <tr>
            <td><span class="table-plate">${v.vehicleNumber}</span></td>
            <td>
              <span style="display:flex;align-items:center;gap:6px">
                <i class="fas ${v.violationIcon}" style="color:${v.violationColor}"></i>
                ${v.violation}
              </span>
            </td>
            <td><span class="table-cam">${v.camera}</span></td>
            <td class="font-mono text-sm">${v.time}</td>
            <td class="glow-text-green font-bold">₹${v.fine.toLocaleString()}</td>
            <td>${renderStatusBadge(v.status)}</td>
            <td class="text-muted text-sm">${v.officer}</td>
            <td>
              <div class="table-actions">
                <button class="btn btn-ghost btn-sm btn-icon" title="View"><i class="fas fa-eye"></i></button>
                <button class="btn btn-primary btn-sm btn-icon" title="Issue Challan"><i class="fas fa-file-invoice"></i></button>
                <button class="btn btn-ghost btn-sm btn-icon" title="Download"><i class="fas fa-download"></i></button>
              </div>
            </td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;

  const renderStatusBadge = (status) => {
    const map = {
      'Pending': 'pending', 'Challan Issued': 'issued',
      'Paid': 'paid', 'Under Review': 'review',
    };
    return `<span class="badge badge-${map[status] || 'pending'}">${status}</span>`;
  };

  // ══════════════════════════════════════════════════════════════
  //  LIVE FEED PAGE
  // ══════════════════════════════════════════════════════════════
  const renderLiveFeed = () => {
    const el = document.getElementById('page-live-feed');
    el.innerHTML = `
      <div class="section-header">
        <div>
          <div class="section-title">Live Camera Feed</div>
          <div class="section-sub">Real-time AI-powered monitoring across ${camerasData.length} cameras</div>
        </div>
        <div class="section-actions">
          <select class="form-control" id="grid-layout" onchange="App.changeGrid(this.value)">
            <option value="4">4×4 Grid</option>
            <option value="3">3×3 Grid</option>
            <option value="2">2×2 Grid</option>
          </select>
          <button class="btn btn-ghost btn-sm"><i class="fas fa-filter"></i> Filter</button>
          <span class="badge badge-online" style="padding:8px 14px">● ${camerasData.filter(c=>c.status==='Online').length} Live</span>
        </div>
      </div>

      <!-- AI Stats Bar -->
      <div class="card mb-6" style="padding:14px 20px">
        <div class="flex items-center gap-4" style="flex-wrap:wrap">
          ${[
            { label:'FPS', value:'30', color: '#00ff88' },
            { label:'Resolution', value:'1080p', color: '#00d4ff' },
            { label:'AI Model', value:'YOLOv9', color: '#7c3aed' },
            { label:'Objects Tracked', value:'2,847', color: '#ffd700' },
            { label:'Detections/min', value:'342', color: '#ff6b35' },
            { label:'Confidence Avg', value:'96.4%', color: '#00ff88' },
            { label:'Latency', value:'42ms', color: '#00d4ff' },
          ].map(s => `
            <div class="flex gap-2 items-center" style="padding:6px 16px;border-right:1px solid var(--border)">
              <span class="text-muted text-xs">${s.label}</span>
              <span class="font-bold font-mono" style="color:${s.color}">${s.value}</span>
            </div>
          `).join('')}
        </div>
      </div>

      <div class="camera-grid" id="live-camera-grid" style="grid-template-columns:repeat(4,1fr)">
        ${renderCameraCards(16)}
      </div>
    `;
    setTimeout(() => startCameraAnimations(), 100);
  };

  // ── Change grid layout ────────────────────────────────────────
  window.App = window.App || {};
  const changeGrid = (cols) => {
    const grid = document.getElementById('live-camera-grid');
    if (grid) grid.style.gridTemplateColumns = `repeat(${cols},1fr)`;
  };

  // ══════════════════════════════════════════════════════════════
  //  AI DETECTION PAGE
  // ══════════════════════════════════════════════════════════════
  const renderAIDetection = () => {
    const el = document.getElementById('page-ai-detection');
    el.innerHTML = `
      <div class="section-header">
        <div>
          <div class="section-title">AI Detection Engine</div>
          <div class="section-sub">Real-time computer vision powered by YOLOv9</div>
        </div>
        <div class="section-actions">
          <span class="badge badge-online" style="padding:8px 14px;font-size:12px">● Model Active</span>
          <button class="btn btn-primary"><i class="fas fa-sync-alt"></i> Retrain Model</button>
        </div>
      </div>

      <!-- Model Performance -->
      <div class="stats-grid mb-6" style="grid-template-columns:repeat(4,1fr)">
        ${[
          { label:'mAP@0.5', value:'97.3%', icon:'fa-bullseye', color:'#00ff88' },
          { label:'Precision', value:'96.8%', icon:'fa-crosshairs', color:'#00d4ff' },
          { label:'Recall', value:'98.1%', icon:'fa-redo', color:'#ffd700' },
          { label:'F1 Score', value:'97.4%', icon:'fa-chart-bar', color:'#ff6b35' },
        ].map(s => `
          <div class="stat-card" style="--accent-start:${s.color};--accent-end:${s.color}">
            <div class="stat-icon-wrap" style="background:${s.color}22;color:${s.color}"><i class="fas ${s.icon}"></i></div>
            <div class="stat-label">${s.label}</div>
            <div class="stat-value" style="background:linear-gradient(135deg,var(--text-primary),${s.color});-webkit-background-clip:text;-webkit-text-fill-color:transparent">${s.value}</div>
          </div>
        `).join('')}
      </div>

      <div class="charts-grid mb-6">
        <div class="chart-col-8">
          <div class="ai-detection-feed">
            <div class="ai-feed-header">
              <div class="ai-live-badge"><div class="ai-live-dot"></div> LIVE INFERENCE — CAM-003</div>
              <div class="ai-model-info">
                <div class="ai-stat"><span class="ai-stat-label">Model: </span><span class="ai-stat-value">YOLOv9-X</span></div>
                <div class="ai-stat"><span class="ai-stat-label">FPS: </span><span class="ai-stat-value">30</span></div>
                <div class="ai-stat"><span class="ai-stat-label">Latency: </span><span class="ai-stat-value">42ms</span></div>
                <div class="ai-stat"><span class="ai-stat-label">GPU: </span><span class="ai-stat-value">87%</span></div>
              </div>
            </div>
            <div style="position:relative;height:340px;background:#060a18">
              <canvas id="ai-main-feed" width="800" height="340" style="width:100%;height:100%"></canvas>
              <!-- Detection boxes will be drawn on canvas -->
            </div>
          </div>
        </div>
        <div class="chart-col-4">
          <div class="card" style="height:100%">
            <div class="card-header"><span class="card-title-main">Detection Classes</span></div>
            ${TrafficData.violationTypes.map((v,i) => `
              <div class="progress-bar-wrap">
                <div class="progress-bar-label">
                  <span style="color:${v.color}"><i class="fas ${v.icon}"></i> ${v.name}</span>
                  <span style="font-weight:700">${Math.floor(Math.random()*200+20)}</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" style="width:${Math.random()*80+20}%;background:${v.color}"></div></div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>

      <!-- Detection Log -->
      <div class="card">
        <div class="card-header">
          <span class="card-title-main">Real-Time Detection Log</span>
          <span class="ai-live-badge"><span class="ai-live-dot"></span>STREAMING</span>
        </div>
        <div class="table-wrapper">
          <table class="data-table">
            <thead><tr><th>#</th><th>Frame ID</th><th>Class</th><th>Confidence</th><th>BBox</th><th>Camera</th><th>Speed</th><th>Time</th></tr></thead>
            <tbody>
              ${Array.from({length:12}, (_,i) => {
                const v = TrafficData.violationTypes[Math.floor(Math.random()*10)];
                const cam = TrafficData.cameraIds[Math.floor(Math.random()*16)];
                return `<tr>
                  <td class="text-muted font-mono text-xs">${i+1}</td>
                  <td class="font-mono text-xs" style="color:var(--neon-blue)">FRM-${String(Math.floor(Math.random()*99999)).padStart(6,'0')}</td>
                  <td><span style="color:${v.color}"><i class="fas ${v.icon}"></i> ${v.name}</span></td>
                  <td><span class="font-mono font-bold" style="color:var(--neon-green)">${(Math.random()*10+88).toFixed(1)}%</span></td>
                  <td class="font-mono text-xs text-muted">[${Math.floor(Math.random()*500)}, ${Math.floor(Math.random()*300)}, ${Math.floor(Math.random()*200+50)}, ${Math.floor(Math.random()*150+30)}]</td>
                  <td><span class="table-cam">${cam}</span></td>
                  <td>${Math.floor(Math.random()*80+20)} km/h</td>
                  <td class="font-mono text-xs text-muted">${TrafficData.randomTime()}</td>
                </tr>`;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
    setTimeout(() => drawAIFeedCanvas(), 200);
  };

  // ── AI Feed Canvas Drawing ────────────────────────────────────
  const drawAIFeedCanvas = () => {
    const canvas = document.getElementById('ai-main-feed');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    const draw = () => {
      if (!document.getElementById('ai-main-feed')) return;
      // Dark background
      ctx.fillStyle = '#060a18';
      ctx.fillRect(0, 0, W, H);

      // Road simulation
      ctx.fillStyle = '#0d1a0d';
      ctx.fillRect(0, H * 0.5, W, H * 0.5);
      ctx.fillStyle = '#111';
      ctx.fillRect(0, H * 0.55, W, H * 0.4);

      // Road markings
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 2;
      ctx.setLineDash([40, 20]);
      ctx.beginPath();
      ctx.moveTo(0, H * 0.65); ctx.lineTo(W, H * 0.65);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw vehicles (simple rectangles)
      const vehicles = [
        { x: 80, y: 180, w: 120, h: 60, color: '#1a4a6e', label: 'Car', conf: 98.2 },
        { x: 280, y: 160, w: 140, h: 70, color: '#4a1a1a', label: 'Truck', conf: 96.7 },
        { x: 500, y: 190, w: 90, h: 50, color: '#2a4a1a', label: 'Motorcycle', conf: 94.1, violation: 'No Helmet' },
        { x: 650, y: 175, w: 115, h: 60, color: '#3a2a5a', label: 'SUV', conf: 97.8 },
      ];

      vehicles.forEach(v => {
        // Vehicle body
        ctx.fillStyle = v.color;
        ctx.fillRect(v.x, v.y, v.w, v.h);

        // Windows
        ctx.fillStyle = 'rgba(100,180,255,0.2)';
        ctx.fillRect(v.x + v.w * 0.15, v.y + v.h * 0.1, v.w * 0.7, v.h * 0.35);

        // Bounding box
        const boxColor = v.violation ? '#ff3366' : '#00ff88';
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 2;
        ctx.strokeRect(v.x - 5, v.y - 30, v.w + 10, v.h + 35);

        // Label
        ctx.fillStyle = boxColor;
        ctx.fillRect(v.x - 5, v.y - 52, v.violation ? 130 : 80, 22);
        ctx.fillStyle = v.violation ? '#fff' : '#000';
        ctx.font = 'bold 10px JetBrains Mono, monospace';
        ctx.fillText(`${v.label} ${v.conf}%`, v.x, v.y - 37);

        if (v.violation) {
          ctx.fillStyle = '#ff3366';
          ctx.fillRect(v.x - 5, v.y - 28, 100, 20);
          ctx.fillStyle = '#fff';
          ctx.font = 'bold 9px Inter, sans-serif';
          ctx.fillText(`⚠ ${v.violation}`, v.x, v.y - 14);
        }

        // Number plate
        ctx.fillStyle = '#ffd700';
        ctx.fillRect(v.x + v.w * 0.2, v.y + v.h - 14, v.w * 0.6, 12);
        ctx.fillStyle = '#000';
        ctx.font = 'bold 7px JetBrains Mono, monospace';
        ctx.fillText(TrafficData.numberPlates().slice(0, 10), v.x + v.w * 0.22, v.y + v.h - 5);
      });

      // Speed overlay
      ctx.fillStyle = 'rgba(0,212,255,0.8)';
      ctx.fillRect(10, 10, 90, 20);
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px JetBrains Mono, monospace';
      ctx.fillText(`AVG: 52 km/h`, 14, 24);

      // Timestamp
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillText(new Date().toLocaleTimeString(), W - 80, H - 10);

      setTimeout(draw, 2000);
    };
    draw();
  };

  // ══════════════════════════════════════════════════════════════
  //  VIOLATIONS PAGE
  // ══════════════════════════════════════════════════════════════
  const renderViolations = () => {
    const el = document.getElementById('page-violations');
    el.innerHTML = `
      <div class="section-header">
        <div>
          <div class="section-title">Violations</div>
          <div class="section-sub">${violationsData.length} violations detected today</div>
        </div>
        <div class="section-actions">
          <button class="btn btn-ghost btn-sm"><i class="fas fa-file-pdf" style="color:#ff3366"></i> PDF</button>
          <button class="btn btn-ghost btn-sm"><i class="fas fa-file-excel" style="color:#00ff88"></i> Excel</button>
          <button class="btn btn-primary btn-sm"><i class="fas fa-plus"></i> Add Manual</button>
        </div>
      </div>

      <!-- Filters -->
      <div class="filter-bar mb-6">
        <input type="text" class="form-control" placeholder="Search vehicle, plate..." style="max-width:240px">
        <select class="form-control">
          <option>All Violations</option>
          ${TrafficData.violationTypes.map(v => `<option>${v.name}</option>`).join('')}
        </select>
        <select class="form-control">
          <option>All Status</option>
          <option>Pending</option>
          <option>Challan Issued</option>
          <option>Paid</option>
          <option>Under Review</option>
        </select>
        <select class="form-control">
          <option>All Cameras</option>
          ${TrafficData.cameraIds.map(c => `<option>${c}</option>`).join('')}
        </select>
        <input type="date" class="form-control" value="2026-08-03">
        <button class="btn btn-primary btn-sm"><i class="fas fa-search"></i> Search</button>
      </div>

      <!-- Violation Cards -->
      <div class="violations-grid mb-6" id="violations-cards">
        ${violationsData.slice(0, 12).map(v => renderViolationCard(v)).join('')}
      </div>

      <!-- Pagination -->
      <div class="pagination">
        ${[1,2,3,4,5].map((p,i) => `<button class="page-btn ${i===0?'active':''}">${p}</button>`).join('')}
        <button class="page-btn">...</button>
        <button class="page-btn">12</button>
        <button class="page-btn"><i class="fas fa-chevron-right"></i></button>
      </div>
    `;
  };

  const renderViolationCard = (v) => `
    <div class="violation-card">
      <div class="vio-image-wrap">
        <canvas class="vio-image-canvas" id="vio-img-${v.id}" width="320" height="130"></canvas>
        <div class="vio-type-badge" style="color:${v.violationColor}">
          <i class="fas ${v.violationIcon}"></i>${v.violation}
        </div>
        <div class="vio-confidence">${v.confidence}%</div>
        <div class="vio-number-plate">${v.vehicleNumber}</div>
      </div>
      <div class="vio-body">
        <div class="vio-meta-grid">
          <div class="vio-meta-item">
            <div class="vio-meta-label">Date</div>
            <div class="vio-meta-value">${v.date}</div>
          </div>
          <div class="vio-meta-item">
            <div class="vio-meta-label">Time</div>
            <div class="vio-meta-value font-mono">${v.time}</div>
          </div>
          <div class="vio-meta-item">
            <div class="vio-meta-label">Camera</div>
            <div class="vio-meta-value" style="color:var(--neon-blue)">${v.camera}</div>
          </div>
          <div class="vio-meta-item">
            <div class="vio-meta-label">Speed</div>
            <div class="vio-meta-value">${v.speed} km/h</div>
          </div>
          <div class="vio-meta-item" style="grid-column:span 2">
            <div class="vio-meta-label">Location</div>
            <div class="vio-meta-value">${v.location}</div>
          </div>
        </div>
        <div class="vio-footer">
          <div class="vio-fine">₹${v.fine.toLocaleString()}<span>Fine Amount</span></div>
          <div class="flex gap-2 items-center">
            ${renderStatusBadge(v.status)}
            <button class="btn btn-primary btn-sm"><i class="fas fa-eye"></i></button>
          </div>
        </div>
      </div>
    </div>
  `;

  // ══════════════════════════════════════════════════════════════
  //  VEHICLES PAGE
  // ══════════════════════════════════════════════════════════════
  const renderVehicles = () => {
    const el = document.getElementById('page-vehicles');
    const typeIcons = { Car:'fa-car', Motorcycle:'fa-motorcycle', Truck:'fa-truck', Bus:'fa-bus', Auto:'fa-taxi', Van:'fa-shuttle-van', SUV:'fa-car-side' };
    const typeColors = { Car:'#00d4ff', Motorcycle:'#ff6b35', Truck:'#ffd700', Bus:'#00ff88', Auto:'#9b59b6', Van:'#ff3366', SUV:'#00d4ff' };
    el.innerHTML = `
      <div class="section-header">
        <div>
          <div class="section-title">Vehicles Registry</div>
          <div class="section-sub">Tracked vehicles in the system</div>
        </div>
        <div class="section-actions">
          <input type="text" class="form-control" placeholder="Search by plate..." style="max-width:200px">
          <button class="btn btn-primary btn-sm"><i class="fas fa-plus"></i> Register Vehicle</button>
        </div>
      </div>
      <div class="vehicle-grid">
        ${vehiclesData.map(v => `
          <div class="vehicle-card">
            <div class="vehicle-icon" style="background:${typeColors[v.type]||'#00d4ff'}22;color:${typeColors[v.type]||'#00d4ff'}">
              <i class="fas ${typeIcons[v.type]||'fa-car'}"></i>
            </div>
            <div class="vehicle-info">
              <div class="vehicle-plate">${v.plate}</div>
              <div class="vehicle-type">${v.type}</div>
              <div class="vehicle-stats">
                <div class="vehicle-stat">Violations: <strong style="color:var(--neon-red)">${v.violations}</strong></div>
                <div class="vehicle-stat">Challans: <strong style="color:var(--neon-orange)">${v.challans}</strong></div>
              </div>
            </div>
            <div class="flex gap-2">
              <button class="btn btn-ghost btn-sm btn-icon"><i class="fas fa-eye"></i></button>
              <button class="btn btn-primary btn-sm btn-icon"><i class="fas fa-file-invoice"></i></button>
            </div>
          </div>
        `).join('')}
      </div>
    `;
  };

  // ══════════════════════════════════════════════════════════════
  //  CHALLANS PAGE
  // ══════════════════════════════════════════════════════════════
  const renderChallans = () => {
    const el = document.getElementById('page-challans');
    el.innerHTML = `
      <div class="section-header">
        <div>
          <div class="section-title">Challans Management</div>
          <div class="section-sub">${challansData.length} challans in system</div>
        </div>
        <div class="section-actions">
          <button class="btn btn-ghost btn-sm"><i class="fas fa-file-pdf" style="color:#ff3366"></i> Export PDF</button>
          <button class="btn btn-success btn-sm"><i class="fas fa-file-excel"></i> Export Excel</button>
        </div>
      </div>
      <div class="filter-bar mb-6">
        <input type="text" class="form-control" placeholder="Search challan, plate..." style="max-width:240px">
        <select class="form-control"><option>All Status</option><option>Pending</option><option>Paid</option><option>Overdue</option></select>
        <input type="date" class="form-control" value="2026-08-03">
        <button class="btn btn-primary btn-sm"><i class="fas fa-search"></i></button>
      </div>
      <div class="card">
        <div class="table-wrapper">
          <table class="data-table">
            <thead><tr><th>Challan ID</th><th>Vehicle No.</th><th>Violation</th><th>Fine</th><th>Issue Date</th><th>Due Date</th><th>Status</th><th>Actions</th></tr></thead>
            <tbody>
              ${challansData.slice(0, 20).map(c => `
                <tr>
                  <td class="font-mono text-sm" style="color:var(--neon-blue)">${c.challanId}</td>
                  <td><span class="table-plate">${c.vehicleNumber}</span></td>
                  <td><i class="fas ${c.violationIcon}" style="color:${c.violationColor}"></i> ${c.violation}</td>
                  <td class="glow-text-green font-bold">₹${c.fine.toLocaleString()}</td>
                  <td class="text-muted text-sm">${c.issueDate}</td>
                  <td class="text-muted text-sm">${c.dueDate}</td>
                  <td>${renderStatusBadge(c.status)}</td>
                  <td>
                    <div class="table-actions">
                      <button class="btn btn-ghost btn-sm btn-icon" title="View"><i class="fas fa-eye"></i></button>
                      <button class="btn btn-primary btn-sm btn-icon" title="Send"><i class="fas fa-paper-plane"></i></button>
                      <button class="btn btn-ghost btn-sm btn-icon" title="Print"><i class="fas fa-print"></i></button>
                    </div>
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
        <div class="pagination" style="padding:16px">
          ${[1,2,3,4,5].map((p,i) => `<button class="page-btn ${i===0?'active':''}">${p}</button>`).join('')}
        </div>
      </div>
    `;
  };

  // ══════════════════════════════════════════════════════════════
  //  ANALYTICS PAGE
  // ══════════════════════════════════════════════════════════════
  const renderAnalytics = () => {
    const el = document.getElementById('page-analytics');
    el.innerHTML = `
      <div class="section-header">
        <div>
          <div class="section-title">Analytics & Insights</div>
          <div class="section-sub">Comprehensive traffic intelligence dashboard</div>
        </div>
        <div class="section-actions">
          <select class="form-control"><option>Last 7 Days</option><option>Last 30 Days</option><option>Last 90 Days</option><option>Custom</option></select>
          <button class="btn btn-primary btn-sm"><i class="fas fa-download"></i> Export</button>
        </div>
      </div>

      <!-- Charts Grid -->
      <div class="charts-grid mb-6">
        <div class="chart-col-12">
          <div class="card">
            <div class="card-header">
              <span class="card-title-main">Daily Traffic & Violations Overview</span>
              <span class="text-muted text-xs">Last 7 days</span>
            </div>
            <div class="chart-container chart-container-lg"><canvas id="daily-bar-chart"></canvas></div>
          </div>
        </div>
      </div>

      <div class="charts-grid mb-6">
        <div class="chart-col-6">
          <div class="card">
            <div class="card-header"><span class="card-title-main">Revenue & Challans Trend</span></div>
            <div class="chart-container chart-container-lg"><canvas id="revenue-chart"></canvas></div>
          </div>
        </div>
        <div class="chart-col-6">
          <div class="card">
            <div class="card-header"><span class="card-title-main">Hourly Traffic Flow</span></div>
            <div class="chart-container chart-container-lg"><canvas id="hourly-chart-2"></canvas></div>
          </div>
        </div>
      </div>

      <div class="charts-grid mb-6">
        <div class="chart-col-4">
          <div class="card">
            <div class="card-header"><span class="card-title-main">Violation Types</span></div>
            <div class="chart-container chart-container-lg"><canvas id="vio-donut-2"></canvas></div>
          </div>
        </div>
        <div class="chart-col-4">
          <div class="card">
            <div class="card-header"><span class="card-title-main">Vehicle Categories</span></div>
            <div class="chart-container chart-container-lg"><canvas id="vehicle-pie"></canvas></div>
          </div>
        </div>
        <div class="chart-col-4">
          <div class="card">
            <div class="card-header"><span class="card-title-main">Violation Trends by Hour</span></div>
            <div class="chart-container chart-container-lg"><canvas id="vio-trend-chart"></canvas></div>
          </div>
        </div>
      </div>

      <!-- Heatmap -->
      <div class="card mb-6">
        <div class="card-header">
          <span class="card-title-main">Traffic Density Heatmap (Hour × Day)</span>
          <span class="text-muted text-xs">Red = High Traffic</span>
        </div>
        <div class="chart-container" style="height:196px">
          <canvas id="heatmap-canvas"></canvas>
        </div>
      </div>

      <!-- Top Locations Table -->
      <div class="card">
        <div class="card-header"><span class="card-title-main">Top Violation Locations</span></div>
        <div class="table-wrapper">
          <table class="data-table">
            <thead><tr><th>#</th><th>Location</th><th>Total Violations</th><th>Cameras</th><th>Top Violation</th><th>Revenue</th><th>Trend</th></tr></thead>
            <tbody>
              ${TrafficData.locations.slice(0, 8).map((loc, i) => {
                const vio = TrafficData.violationTypes[Math.floor(Math.random() * 10)];
                const count = Math.floor(Math.random() * 200 + 50);
                return `<tr>
                  <td class="text-muted font-bold">${i+1}</td>
                  <td class="font-bold">${loc}</td>
                  <td><span class="glow-text-red font-bold">${count}</span></td>
                  <td class="table-cam">${Math.floor(Math.random()*5+2)}</td>
                  <td><i class="fas ${vio.icon}" style="color:${vio.color}"></i> ${vio.name}</td>
                  <td class="glow-text-green">₹${(count * Math.floor(Math.random()*2000+500)).toLocaleString()}</td>
                  <td class="${Math.random()>0.5?'trend-up':'trend-down'}">
                    <i class="fas ${Math.random()>0.5?'fa-arrow-up':'fa-arrow-down'}"></i> ${Math.floor(Math.random()*20+2)}%
                  </td>
                </tr>`;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
    setTimeout(() => {
      Charts.initDailyTraffic('daily-bar-chart');
      Charts.initRevenueChart('revenue-chart');
      Charts.initHourlyTraffic('hourly-chart-2');
      Charts.initViolationDonut('vio-donut-2');
      Charts.initVehiclePie('vehicle-pie');
      Charts.initViolationTrend('vio-trend-chart');
      Charts.initHeatmap('heatmap-canvas');
    }, 100);
  };

  // ══════════════════════════════════════════════════════════════
  //  REPORTS PAGE
  // ══════════════════════════════════════════════════════════════
  const renderReports = () => {
    const el = document.getElementById('page-reports');
    el.innerHTML = `
      <div class="section-header">
        <div>
          <div class="section-title">Reports Center</div>
          <div class="section-sub">Generate, download, and schedule reports</div>
        </div>
      </div>

      <!-- Report Types -->
      <div class="reports-grid mb-6">
        ${[
          { icon:'📊', name:'Daily Report',    desc:'Full daily violations summary',  color:'#00d4ff', fmt:'PDF/Excel' },
          { icon:'📅', name:'Weekly Report',   desc:'7-day traffic analysis',         color:'#00ff88', fmt:'PDF/Excel' },
          { icon:'📈', name:'Monthly Report',  desc:'Monthly KPIs and trends',        color:'#ffd700', fmt:'PDF' },
          { icon:'💰', name:'Revenue Report',  desc:'Challan revenue analytics',      color:'#ff6b35', fmt:'Excel/CSV' },
          { icon:'📷', name:'Camera Report',   desc:'Camera performance & health',    color:'#9b59b6', fmt:'PDF' },
          { icon:'🚨', name:'Violation Report',desc:'Detailed violation breakdown',   color:'#ff3366', fmt:'PDF/CSV' },
          { icon:'🤖', name:'AI Performance',  desc:'Model accuracy & metrics',       color:'#00d4ff', fmt:'PDF' },
          { icon:'👤', name:'Officer Report',  desc:'Officer activity summary',       color:'#00ff88', fmt:'PDF/Excel' },
        ].map(r => `
          <div class="report-card">
            <div class="report-icon">${r.icon}</div>
            <div class="report-name">${r.name}</div>
            <div class="report-desc">${r.desc}</div>
            <span class="report-export-badge" style="background:${r.color}22;color:${r.color};border:1px solid ${r.color}44">${r.fmt}</span>
            <div style="margin-top:12px">
              <button class="btn btn-primary btn-sm w-full" style="justify-content:center"><i class="fas fa-download"></i> Generate</button>
            </div>
          </div>
        `).join('')}
      </div>

      <!-- Custom Date Filter -->
      <div class="card mb-6">
        <div class="card-header"><span class="card-title-main">Custom Date Range Report</span></div>
        <div class="flex gap-3 items-center flex-wrap">
          <div class="form-group">
            <label class="form-label">Start Date</label>
            <input type="date" class="form-control" value="2026-07-01">
          </div>
          <div class="form-group">
            <label class="form-label">End Date</label>
            <input type="date" class="form-control" value="2026-08-03">
          </div>
          <div class="form-group">
            <label class="form-label">Report Type</label>
            <select class="form-control">
              <option>Violations Summary</option>
              <option>Revenue Analysis</option>
              <option>Camera Performance</option>
              <option>Complete Report</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Format</label>
            <select class="form-control">
              <option>PDF</option>
              <option>Excel (.xlsx)</option>
              <option>CSV</option>
              <option>JSON</option>
            </select>
          </div>
          <div style="margin-top:20px;display:flex;gap:8px">
            <button class="btn btn-primary"><i class="fas fa-eye"></i> Preview</button>
            <button class="btn btn-success"><i class="fas fa-download"></i> Download</button>
          </div>
        </div>
      </div>

      <!-- Scheduled Reports -->
      <div class="card">
        <div class="card-header">
          <span class="card-title-main">Scheduled Reports</span>
          <button class="btn btn-primary btn-sm"><i class="fas fa-plus"></i> Schedule New</button>
        </div>
        <div class="table-wrapper">
          <table class="data-table">
            <thead><tr><th>Report Name</th><th>Frequency</th><th>Format</th><th>Recipients</th><th>Last Run</th><th>Next Run</th><th>Status</th><th>Actions</th></tr></thead>
            <tbody>
              ${[
                ['Daily Violations', 'Daily 00:00', 'PDF + Excel', '5 users', '2026-08-02', '2026-08-04', 'Active'],
                ['Weekly Summary', 'Every Monday', 'PDF', '8 users', '2026-07-28', '2026-08-10', 'Active'],
                ['Monthly Revenue', 'Monthly 1st', 'Excel', '3 users', '2026-08-01', '2026-09-01', 'Active'],
                ['Camera Health', 'Daily 08:00', 'PDF', '2 users', '2026-08-03', '2026-08-04', 'Paused'],
              ].map(r => `<tr>
                <td class="font-bold">${r[0]}</td>
                <td class="text-muted text-sm">${r[1]}</td>
                <td style="color:var(--neon-blue)">${r[2]}</td>
                <td class="text-muted text-sm">${r[3]}</td>
                <td class="font-mono text-xs text-muted">${r[4]}</td>
                <td class="font-mono text-xs text-muted">${r[5]}</td>
                <td><span class="badge ${r[6]==='Active'?'badge-online':'badge-offline'}">${r[6]}</span></td>
                <td><div class="table-actions">
                  <button class="btn btn-ghost btn-sm btn-icon"><i class="fas fa-edit"></i></button>
                  <button class="btn btn-ghost btn-sm btn-icon"><i class="fas fa-play"></i></button>
                  <button class="btn btn-ghost btn-sm btn-icon"><i class="fas fa-trash" style="color:var(--neon-red)"></i></button>
                </div></td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;
  };

  // ══════════════════════════════════════════════════════════════
  //  CAMERAS PAGE
  // ══════════════════════════════════════════════════════════════
  const renderCameras = () => {
    const el = document.getElementById('page-cameras');
    el.innerHTML = `
      <div class="section-header">
        <div>
          <div class="section-title">Camera Management</div>
          <div class="section-sub">${camerasData.filter(c=>c.status==='Online').length} online / ${camerasData.length} total cameras</div>
        </div>
        <div class="section-actions">
          <button class="btn btn-ghost btn-sm"><i class="fas fa-sync-alt"></i> Refresh</button>
          <button class="btn btn-primary btn-sm"><i class="fas fa-plus"></i> Add Camera</button>
        </div>
      </div>

      <!-- Summary Stats -->
      <div class="stats-grid mb-6" style="grid-template-columns:repeat(4,1fr)">
        ${[
          { label:'Total Cameras', value:camerasData.length, icon:'fa-video', color:'#00d4ff' },
          { label:'Online', value:camerasData.filter(c=>c.status==='Online').length, icon:'fa-check-circle', color:'#00ff88' },
          { label:'Offline', value:camerasData.filter(c=>c.status!=='Online').length, icon:'fa-times-circle', color:'#ff3366' },
          { label:'Avg Uptime', value:'96.4%', icon:'fa-chart-line', color:'#ffd700' },
        ].map(s => `
          <div class="stat-card" style="--accent-start:${s.color};--accent-end:${s.color}">
            <div class="stat-icon-wrap" style="background:${s.color}22;color:${s.color}"><i class="fas ${s.icon}"></i></div>
            <div class="stat-label">${s.label}</div>
            <div class="stat-value" style="background:linear-gradient(135deg,var(--text-primary),${s.color});-webkit-background-clip:text;-webkit-text-fill-color:transparent">${s.value}</div>
          </div>
        `).join('')}
      </div>

      <div class="camera-grid-mgmt">
        ${camerasData.map(cam => {
          const isOnline = cam.status === 'Online';
          const color = isOnline ? 'var(--neon-green)' : 'var(--neon-red)';
          const health = parseFloat(cam.uptime);
          return `
            <div class="camera-mgmt-card" style="--cam-status-color:${color}">
              <div class="cam-mgmt-header">
                <div>
                  <div class="cam-mgmt-id">${cam.id}</div>
                  <div class="cam-mgmt-loc"><i class="fas fa-map-marker-alt"></i> ${cam.location}</div>
                </div>
                <span class="badge ${isOnline?'badge-online':'badge-offline'}">${cam.status}</span>
              </div>
              <div class="cam-specs">
                <div class="cam-spec"><div class="cam-spec-label">Resolution</div><div class="cam-spec-value">${cam.resolution}</div></div>
                <div class="cam-spec"><div class="cam-spec-label">FPS</div><div class="cam-spec-value">${cam.fps}</div></div>
                <div class="cam-spec"><div class="cam-spec-label">AI Model</div><div class="cam-spec-value" style="color:var(--neon-blue)">${cam.aiModel}</div></div>
                <div class="cam-spec"><div class="cam-spec-label">Detections</div><div class="cam-spec-value">${cam.detections.toLocaleString()}</div></div>
                <div class="cam-spec"><div class="cam-spec-label">Last Maintenance</div><div class="cam-spec-value text-sm" style="color:var(--text-muted)">${cam.lastMaintenance}</div></div>
                <div class="cam-spec"><div class="cam-spec-label">Uptime</div><div class="cam-spec-value" style="color:${color}">${cam.uptime}</div></div>
              </div>
              <div class="health-bar"><div class="health-fill" style="width:${health}%"></div></div>
              <div style="display:flex;gap:8px;margin-top:12px">
                <button class="btn btn-ghost btn-sm" style="flex:1;justify-content:center"><i class="fas fa-eye"></i> View</button>
                <button class="btn btn-primary btn-sm" style="flex:1;justify-content:center"><i class="fas fa-cog"></i> Config</button>
                ${!isOnline ? `<button class="btn btn-success btn-sm"><i class="fas fa-power-off"></i></button>` : `<button class="btn btn-danger btn-sm"><i class="fas fa-ban"></i></button>`}
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  };

  // ══════════════════════════════════════════════════════════════
  //  USERS PAGE
  // ══════════════════════════════════════════════════════════════
  const renderUsers = () => {
    const el = document.getElementById('page-users');
    const roleColors = { Admin: '#ff3366', Officer: '#00d4ff', Operator: '#00ff88', Viewer: '#ffd700' };
    el.innerHTML = `
      <div class="section-header">
        <div>
          <div class="section-title">User Management</div>
          <div class="section-sub">${usersData.length} users in the system</div>
        </div>
        <div class="section-actions">
          <button class="btn btn-ghost btn-sm"><i class="fas fa-filter"></i> Filter</button>
          <button class="btn btn-primary btn-sm"><i class="fas fa-user-plus"></i> Add User</button>
        </div>
      </div>

      <!-- Role Summary -->
      <div class="stats-grid mb-6" style="grid-template-columns:repeat(4,1fr)">
        ${Object.entries(roleColors).map(([role, color]) => {
          const count = usersData.filter(u => u.role === role).length;
          return `<div class="stat-card" style="--accent-start:${color};--accent-end:${color}">
            <div class="stat-icon-wrap" style="background:${color}22;color:${color}"><i class="fas fa-user-shield"></i></div>
            <div class="stat-label">${role}s</div>
            <div class="stat-value" style="background:linear-gradient(135deg,var(--text-primary),${color});-webkit-background-clip:text;-webkit-text-fill-color:transparent">${count}</div>
          </div>`;
        }).join('')}
      </div>

      <div class="users-grid">
        ${usersData.map(u => {
          const initials = u.name.split(' ').map(n=>n[0]).join('');
          const color = roleColors[u.role] || '#00d4ff';
          return `
            <div class="user-card">
              <div class="user-avatar-ring" style="background:${color}22;color:${color};--role-color:${color}">
                ${initials}
              </div>
              <div class="user-name">${u.name}</div>
              <div class="user-email">${u.email}</div>
              <div class="user-meta">
                <span class="badge badge-${u.role.toLowerCase()}">${u.role}</span>
                <span class="user-status-dot">
                  <span class="status-dot" style="background:${u.status==='Active'?'var(--neon-green)':'var(--neon-red)'}"></span>
                  ${u.status}
                </span>
              </div>
              <div style="margin-top:12px;font-size:11px;color:var(--text-muted)">Last login: ${u.lastLogin}</div>
              <div style="display:flex;gap:8px;margin-top:12px">
                <button class="btn btn-ghost btn-sm" style="flex:1;justify-content:center"><i class="fas fa-edit"></i> Edit</button>
                <button class="btn btn-ghost btn-sm btn-icon" title="Permissions"><i class="fas fa-key"></i></button>
                <button class="btn btn-ghost btn-sm btn-icon" style="color:var(--neon-red)" title="Delete"><i class="fas fa-trash"></i></button>
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;
  };

  // ══════════════════════════════════════════════════════════════
  //  SETTINGS PAGE
  // ══════════════════════════════════════════════════════════════
  const renderSettings = () => {
    const el = document.getElementById('page-settings');
    el.innerHTML = `
      <div class="section-header">
        <div>
          <div class="section-title">System Settings</div>
          <div class="section-sub">Configure AI, cameras, notifications, and more</div>
        </div>
        <div class="section-actions">
          <button class="btn btn-ghost btn-sm"><i class="fas fa-undo"></i> Reset Defaults</button>
          <button class="btn btn-primary btn-sm"><i class="fas fa-save"></i> Save Changes</button>
        </div>
      </div>

      <div class="settings-grid">
        <!-- AI Configuration -->
        <div class="card">
          <div class="settings-section-title">🤖 AI Configuration</div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Detection Confidence Threshold</div>
              <div class="setting-desc">Minimum confidence for a valid detection</div>
            </div>
            <div class="range-wrap">
              <input type="range" class="range-slider" min="70" max="99" value="85" oninput="this.nextElementSibling.textContent=this.value+'%'">
              <span class="range-value">85%</span>
            </div>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">AI Model Version</div>
              <div class="setting-desc">Currently active detection model</div>
            </div>
            <select class="form-control" style="max-width:160px">
              <option>YOLOv9-X</option><option>YOLOv8-L</option><option>RT-DETR</option>
            </select>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Auto Model Update</div>
              <div class="setting-desc">Automatically update AI models</div>
            </div>
            <label class="toggle-switch"><input type="checkbox" checked><span class="toggle-slider"></span></label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Speed Detection</div>
              <div class="setting-desc">Enable vehicle speed monitoring</div>
            </div>
            <label class="toggle-switch"><input type="checkbox" checked><span class="toggle-slider"></span></label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Speed Threshold (km/h)</div>
              <div class="setting-desc">Alert trigger speed limit</div>
            </div>
            <div class="range-wrap">
              <input type="range" class="range-slider" min="30" max="120" value="60" oninput="this.nextElementSibling.textContent=this.value+' km/h'">
              <span class="range-value">60 km/h</span>
            </div>
          </div>
        </div>

        <!-- Camera Settings -->
        <div class="card">
          <div class="settings-section-title">📷 Camera Settings</div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Default Resolution</div>
              <div class="setting-desc">Video recording resolution</div>
            </div>
            <select class="form-control" style="max-width:120px"><option>4K</option><option>1080p</option><option>720p</option></select>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Frame Rate (FPS)</div>
              <div class="setting-desc">Camera capture frame rate</div>
            </div>
            <select class="form-control" style="max-width:120px"><option>60 FPS</option><option>30 FPS</option><option>24 FPS</option></select>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Night Vision Mode</div>
              <div class="setting-desc">Auto enable IR at low light</div>
            </div>
            <label class="toggle-switch"><input type="checkbox" checked><span class="toggle-slider"></span></label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Recording Duration (days)</div>
              <div class="setting-desc">Video retention period</div>
            </div>
            <div class="range-wrap">
              <input type="range" class="range-slider" min="7" max="90" value="30" oninput="this.nextElementSibling.textContent=this.value+' days'">
              <span class="range-value">30 days</span>
            </div>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Auto Health Check</div>
              <div class="setting-desc">Scheduled camera diagnostics</div>
            </div>
            <label class="toggle-switch"><input type="checkbox" checked><span class="toggle-slider"></span></label>
          </div>
        </div>

        <!-- Notification Preferences -->
        <div class="card">
          <div class="settings-section-title">🔔 Notification Preferences</div>
          ${[
            ['Email Alerts', 'Send violation alerts via email', true],
            ['SMS Alerts', 'Send SMS for critical violations', true],
            ['Push Notifications', 'Browser push notifications', true],
            ['WhatsApp Alerts', 'WhatsApp integration alerts', false],
            ['Dashboard Alerts', 'In-app notification panel', true],
            ['Weekly Digest', 'Weekly email summary', true],
          ].map(([name, desc, checked]) => `
            <div class="setting-row">
              <div class="setting-info">
                <div class="setting-name">${name}</div>
                <div class="setting-desc">${desc}</div>
              </div>
              <label class="toggle-switch"><input type="checkbox" ${checked?'checked':''}><span class="toggle-slider"></span></label>
            </div>
          `).join('')}
        </div>

        <!-- API Configuration -->
        <div class="card">
          <div class="settings-section-title">🔌 API Configuration</div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">API Key</div>
              <div class="setting-desc">System integration API key</div>
            </div>
            <button class="btn btn-ghost btn-sm"><i class="fas fa-key"></i> Regenerate</button>
          </div>
          <div style="margin:12px 0">
            <input type="text" class="form-control w-full" value="sk_live_xxxxxxxx•••••••••••••••••" style="font-family:'JetBrains Mono',monospace">
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Webhook URL</div>
              <div class="setting-desc">External system integration</div>
            </div>
          </div>
          <div style="margin:8px 0">
            <input type="url" class="form-control w-full" placeholder="https://your-server.com/webhook">
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Rate Limit</div>
              <div class="setting-desc">API calls per minute</div>
            </div>
            <div class="range-wrap">
              <input type="range" class="range-slider" min="100" max="5000" value="1000" step="100" oninput="this.nextElementSibling.textContent=this.value+'/min'">
              <span class="range-value">1000/min</span>
            </div>
          </div>
          <div style="margin-top:16px;display:flex;gap:8px">
            <button class="btn btn-primary" style="flex:1;justify-content:center"><i class="fas fa-save"></i> Save API Config</button>
            <button class="btn btn-ghost"><i class="fas fa-vial"></i> Test</button>
          </div>
        </div>

        <!-- Appearance -->
        <div class="card">
          <div class="settings-section-title">🎨 Appearance</div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Dark Mode</div>
              <div class="setting-desc">Toggle dark/light theme</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" checked id="theme-toggle" onchange="App.toggleTheme()">
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Glassmorphism Effects</div>
              <div class="setting-desc">Card blur and transparency</div>
            </div>
            <label class="toggle-switch"><input type="checkbox" checked><span class="toggle-slider"></span></label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Animations</div>
              <div class="setting-desc">Smooth transitions & micro-animations</div>
            </div>
            <label class="toggle-switch"><input type="checkbox" checked><span class="toggle-slider"></span></label>
          </div>
          <div class="setting-row">
            <div class="setting-info">
              <div class="setting-name">Compact Mode</div>
              <div class="setting-desc">Reduce spacing for more data</div>
            </div>
            <label class="toggle-switch"><input type="checkbox"><span class="toggle-slider"></span></label>
          </div>
        </div>

        <!-- System Info -->
        <div class="card">
          <div class="settings-section-title">ℹ️ System Information</div>
          ${[
            ['Version', 'v4.2.1 (Build 20260803)'],
            ['Server', 'traffic-ai-prod-01.gov.in'],
            ['Database', 'PostgreSQL 16.1'],
            ['AI Engine', 'YOLOv9-X (PyTorch 2.2)'],
            ['GPU', 'NVIDIA A100 80GB'],
            ['Uptime', '47 days, 6h 23m'],
            ['Last Backup', '2026-08-03 02:00 AM'],
            ['License', 'Enterprise — City Traffic Authority'],
          ].map(([k,v]) => `
            <div class="setting-row">
              <div class="setting-name">${k}</div>
              <div class="font-mono text-sm" style="color:var(--neon-blue)">${v}</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  };

  // ══════════════════════════════════════════════════════════════
  //  CAMERA CANVAS ANIMATIONS
  // ══════════════════════════════════════════════════════════════
  const startCameraAnimations = () => {
    if (cameraAnimInterval) clearInterval(cameraAnimInterval);
    const animateCamera = (cam, camId) => {
      const canvas = document.getElementById(`feed-${camId}`);
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      if (cam.status !== 'Online') {
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = 'rgba(255,51,102,0.3)';
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = '#ff3366';
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('CAMERA OFFLINE', W/2, H/2);
        ctx.textAlign = 'left';
        return;
      }

      let frame = 0;
      const vehicles = Array.from({length: Math.floor(Math.random()*3+1)}, () => ({
        x: Math.random() * (W - 60), y: H * 0.45 + Math.random() * (H * 0.35),
        w: 40 + Math.random() * 40, h: 20 + Math.random() * 15,
        dx: (Math.random() - 0.5) * 2,
        color: `hsl(${Math.floor(Math.random()*360)},50%,${25+Math.floor(Math.random()*20)}%)`,
        isViolation: Math.random() < 0.25,
      }));

      const drawFrame = () => {
        if (!document.getElementById(`feed-${camId}`)) return;
        frame++;
        // Background
        ctx.fillStyle = '#0a0f1a';
        ctx.fillRect(0, 0, W, H);
        // Ground
        ctx.fillStyle = '#0d1a10';
        ctx.fillRect(0, H * 0.45, W, H * 0.55);
        ctx.fillStyle = '#111820';
        ctx.fillRect(0, H * 0.5, W, H * 0.5);
        // Road lines
        ctx.strokeStyle = '#2a3a2a';
        ctx.lineWidth = 1; ctx.setLineDash([20, 10]);
        for (let y = H * 0.58; y < H; y += H * 0.2) {
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
        }
        ctx.setLineDash([]);
        // Buildings (bg)
        ctx.fillStyle = '#0d1420';
        for (let i = 0; i < 6; i++) {
          const bH = 30 + Math.random() * 40;
          ctx.fillRect(i * (W/6), H * 0.45 - bH, W/6 - 2, bH);
        }
        // Vehicles
        vehicles.forEach(v => {
          v.x += v.dx;
          if (v.x < 0 || v.x + v.w > W) v.dx *= -1;
          ctx.fillStyle = v.color;
          ctx.fillRect(v.x, v.y, v.w, v.h);
          ctx.fillStyle = 'rgba(100,200,255,0.15)';
          ctx.fillRect(v.x + v.w * 0.1, v.y + 2, v.w * 0.8, v.h * 0.4);
          // Box
          const boxColor = v.isViolation ? '#ff3366' : '#00ff88';
          ctx.strokeStyle = boxColor;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(v.x - 3, v.y - 18, v.w + 6, v.h + 21);
          // Label
          ctx.fillStyle = boxColor;
          ctx.fillRect(v.x - 3, v.y - 28, 60, 10);
          ctx.fillStyle = v.isViolation ? '#fff' : '#000';
          ctx.font = '6px JetBrains Mono, monospace';
          ctx.fillText(v.isViolation ? `⚠ VIOLATION` : `VEHICLE`, v.x - 1, v.y - 20);
          // Plate
          ctx.fillStyle = '#ffd700';
          ctx.fillRect(v.x + v.w*0.15, v.y + v.h - 5, v.w * 0.7, 7);
          ctx.fillStyle = '#000';
          ctx.font = '5px JetBrains Mono, monospace';
          ctx.fillText('DL 5C AB', v.x + v.w*0.18, v.y + v.h + 0);
        });
        // Timestamp
        const ts = new Date().toLocaleTimeString();
        const tsEl = document.getElementById(`ts-${camId}`);
        if (tsEl) tsEl.textContent = ts;
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(0, H - 14, 80, 14);
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = '7px JetBrains Mono, monospace';
        ctx.fillText(ts, 3, H - 4);

        setTimeout(drawFrame, 500);
      };
      drawFrame();
    };

    camerasData.slice(0, 16).forEach(cam => {
      animateCamera(cam, cam.id);
    });
  };

  // ══════════════════════════════════════════════════════════════
  //  ACTIVITY FEED AUTO-UPDATE
  // ══════════════════════════════════════════════════════════════
  const startActivityFeed = () => {
    if (activityInterval) clearInterval(activityInterval);
    activityInterval = setInterval(() => {
      const feed = document.getElementById('activity-feed');
      if (!feed) { clearInterval(activityInterval); return; }
      const v = violationsData[Math.floor(Math.random() * violationsData.length)];
      const icons = [
        { icon: 'fa-exclamation-circle', bg: 'rgba(255,51,102,0.15)', color: '#ff3366' },
        { icon: 'fa-video', bg: 'rgba(0,212,255,0.15)', color: '#00d4ff' },
        { icon: 'fa-car', bg: 'rgba(0,255,136,0.15)', color: '#00ff88' },
        { icon: 'fa-file-invoice', bg: 'rgba(255,107,53,0.15)', color: '#ff6b35' },
      ];
      const ic = icons[Math.floor(Math.random() * icons.length)];
      const item = document.createElement('div');
      item.className = 'activity-item';
      item.innerHTML = `
        <div class="activity-icon" style="background:${ic.bg};color:${ic.color}"><i class="fas ${ic.icon}"></i></div>
        <div class="activity-text">
          <div class="activity-main">${v.violation} — <span class="table-plate">${v.vehicleNumber}</span></div>
          <div class="activity-sub">${v.location} · ${v.camera}</div>
        </div>
        <div class="activity-time">${new Date().toLocaleTimeString()}</div>
      `;
      feed.insertBefore(item, feed.firstChild);
      if (feed.children.length > 15) feed.removeChild(feed.lastChild);
    }, 3000);
  };

  // ══════════════════════════════════════════════════════════════
  //  ANIMATED COUNTERS
  // ══════════════════════════════════════════════════════════════
  const animateCounters = () => {
    if (statsCounterDone) return;
    document.querySelectorAll('.counter').forEach(el => {
      const target = el.dataset.target;
      if (!target || isNaN(target.replace(/[₹%,]/g, ''))) return;
      const raw = parseFloat(target.replace(/[₹%,]/g, ''));
      const prefix = target.startsWith('₹') ? '₹' : '';
      const suffix = target.endsWith('%') ? '%' : '';
      let current = 0;
      const duration = 1500;
      const step = raw / (duration / 16);
      const update = () => {
        current = Math.min(current + step, raw);
        if (prefix === '₹') el.textContent = prefix + formatNum(Math.floor(current));
        else if (suffix === '%') el.textContent = current.toFixed(1) + suffix;
        else el.textContent = prefix + Math.floor(current).toLocaleString() + suffix;
        if (current < raw) requestAnimationFrame(update);
      };
      requestAnimationFrame(update);
    });
    statsCounterDone = true;
  };

  // ══════════════════════════════════════════════════════════════
  //  THEME TOGGLE
  // ══════════════════════════════════════════════════════════════
  const toggleTheme = () => {
    lightMode = !lightMode;
    document.body.classList.toggle('light-mode', lightMode);
  };

  // ── Helpers ───────────────────────────────────────────────────
  const formatNum = (n) => {
    if (n >= 10000000) return (n/10000000).toFixed(1) + 'Cr';
    if (n >= 100000) return (n/100000).toFixed(1) + 'L';
    if (n >= 1000) return (n/1000).toFixed(1) + 'K';
    return n.toString();
  };

  // ══════════════════════════════════════════════════════════════
  //  INIT
  // ══════════════════════════════════════════════════════════════
  const init = () => {
    initData();
    renderSidebar();

    // Mobile sidebar
    document.getElementById('mobile-menu-btn')?.addEventListener('click', () => {
      document.getElementById('sidebar')?.classList.toggle('mobile-open');
      document.getElementById('sidebar-overlay')?.classList.toggle('show');
    });
    document.getElementById('sidebar-overlay')?.addEventListener('click', () => {
      document.getElementById('sidebar')?.classList.remove('mobile-open');
      document.getElementById('sidebar-overlay')?.classList.remove('show');
    });

    // Sidebar collapse
    document.getElementById('sidebar-collapse-btn')?.addEventListener('click', () => {
      document.getElementById('sidebar')?.classList.toggle('collapsed');
    });

    // Notification system
    NotificationSystem.init();

    // Navigate to dashboard
    navigate('dashboard');
  };

  return { init, navigate, renderPage, toggleTheme, changeGrid };
})();

window.App = App;
document.addEventListener('DOMContentLoaded', () => App.init());
