// ============================================================
// Smart Traffic Crime Detection — Notifications Module
// ============================================================

const NotificationSystem = (() => {
  let notifications = [];
  let unreadCount = 0;
  let dropdownOpen = false;

  const notificationTypes = [
    { type: 'alert', icon: 'fa-exclamation-circle', color: '#ff3366', label: 'Alert' },
    { type: 'detection', icon: 'fa-eye', color: '#00d4ff', label: 'Detection' },
    { type: 'camera', icon: 'fa-video', color: '#ffd700', label: 'Camera' },
    { type: 'system', icon: 'fa-server', color: '#00ff88', label: 'System' },
    { type: 'challan', icon: 'fa-file-invoice', color: '#ff6b35', label: 'Challan' },
  ];

  const messages = [
    'Red light jumping detected at MG Road — CAM-003',
    'Overspeeding: DL 5C AB 1234 @ 89 km/h — CAM-007',
    'Camera CAM-012 went offline at Nehru Place',
    'No helmet detected — Motorcycle rider — CAM-002',
    'AI Model updated to YOLOv9 — Accuracy: 97.8%',
    'Challan issued: MH 02 KL 7890 — Seatbelt Violation',
    'Triple riding detected at Brigade Road — CAM-009',
    'System health check passed — All systems nominal',
    'Wrong direction — KA 01 MN 2345 — CAM-005',
    'Mobile phone usage detected — UP 16 CD 4567',
    'Camera CAM-008 reconnected — Signal restored',
    'Daily report generated — 1,247 violations detected',
  ];

  const generateNotification = () => {
    const type = notificationTypes[Math.floor(Math.random() * notificationTypes.length)];
    const message = messages[Math.floor(Math.random() * messages.length)];
    return {
      id: Date.now() + Math.random(),
      type: type.type,
      icon: type.icon,
      color: type.color,
      label: type.label,
      message,
      time: new Date().toLocaleTimeString(),
      read: false,
    };
  };

  const addNotification = (notif) => {
    notifications.unshift(notif);
    if (notifications.length > 20) notifications.pop();
    unreadCount++;
    updateBadge();
    renderDropdown();
    showToast(notif);
  };

  const updateBadge = () => {
    const badge = document.getElementById('notif-badge');
    if (badge) {
      badge.textContent = unreadCount > 9 ? '9+' : unreadCount;
      badge.style.display = unreadCount > 0 ? 'flex' : 'none';
    }
  };

  const markAllRead = () => {
    notifications.forEach(n => n.read = true);
    unreadCount = 0;
    updateBadge();
    renderDropdown();
  };

  const renderDropdown = () => {
    const list = document.getElementById('notif-list');
    if (!list) return;
    if (notifications.length === 0) {
      list.innerHTML = `
        <div class="notif-empty">
          <i class="fas fa-bell-slash"></i>
          <p>No notifications</p>
        </div>`;
      return;
    }
    list.innerHTML = notifications.slice(0, 8).map(n => `
      <div class="notif-item ${n.read ? 'read' : ''}" style="border-left: 3px solid ${n.color}">
        <div class="notif-icon" style="color: ${n.color}">
          <i class="fas ${n.icon}"></i>
        </div>
        <div class="notif-content">
          <span class="notif-label" style="color: ${n.color}">${n.label}</span>
          <p class="notif-msg">${n.message}</p>
          <span class="notif-time">${n.time}</span>
        </div>
        ${!n.read ? '<div class="notif-dot"></div>' : ''}
      </div>
    `).join('');
  };

  const showToast = (notif) => {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'toast-item';
    toast.style.borderLeft = `3px solid ${notif.color}`;
    toast.innerHTML = `
      <div class="toast-icon" style="color: ${notif.color}"><i class="fas ${notif.icon}"></i></div>
      <div class="toast-body">
        <span class="toast-label" style="color: ${notif.color}">${notif.label}</span>
        <p class="toast-msg">${notif.message}</p>
      </div>
      <button class="toast-close" onclick="this.parentElement.remove()"><i class="fas fa-times"></i></button>
    `;
    container.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => toast.remove(), 400);
    }, 4000);
  };

  const toggleDropdown = () => {
    const dropdown = document.getElementById('notif-dropdown');
    if (!dropdown) return;
    dropdownOpen = !dropdownOpen;
    dropdown.classList.toggle('show', dropdownOpen);
    if (dropdownOpen) markAllRead();
  };

  const init = () => {
    // Pre-populate with initial notifications
    for (let i = 0; i < 5; i++) {
      notifications.push({ ...generateNotification(), read: i > 1 });
    }
    unreadCount = 2;
    updateBadge();
    renderDropdown();

    // Auto-generate every 8–15 seconds
    const autoNotif = () => {
      addNotification(generateNotification());
      setTimeout(autoNotif, 8000 + Math.random() * 7000);
    };
    setTimeout(autoNotif, 5000);

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.notif-wrapper')) {
        const dropdown = document.getElementById('notif-dropdown');
        if (dropdown) {
          dropdown.classList.remove('show');
          dropdownOpen = false;
        }
      }
    });
  };

  return { init, toggleDropdown, markAllRead };
})();

window.NotificationSystem = NotificationSystem;
