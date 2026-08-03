// ============================================================
// Smart Traffic Crime Detection — Data Generator
// ============================================================

const TrafficData = (() => {
  const violationTypes = [
    { name: 'Helmet Violation', icon: 'fa-hard-hat', color: '#ff6b35', fine: 500 },
    { name: 'Seatbelt Violation', icon: 'fa-car-crash', color: '#ff3366', fine: 1000 },
    { name: 'Triple Riding', icon: 'fa-motorcycle', color: '#ffd700', fine: 2000 },
    { name: 'Wrong Direction', icon: 'fa-exclamation-triangle', color: '#ff3366', fine: 5000 },
    { name: 'Red Light Jumping', icon: 'fa-traffic-light', color: '#ff3366', fine: 3000 },
    { name: 'Overspeeding', icon: 'fa-tachometer-alt', color: '#ff6b35', fine: 2500 },
    { name: 'Illegal Parking', icon: 'fa-parking', color: '#ffd700', fine: 1500 },
    { name: 'Mobile Phone Usage', icon: 'fa-mobile-alt', color: '#ff6b35', fine: 2000 },
    { name: 'No Number Plate', icon: 'fa-id-card', color: '#ff3366', fine: 5000 },
    { name: 'Expired Registration', icon: 'fa-calendar-times', color: '#ff6b35', fine: 3500 },
  ];

  const locations = [
    'MG Road Junction', 'Connaught Place', 'Brigade Road', 'Nehru Place',
    'Karol Bagh Chowk', 'Lajpat Nagar', 'Saket Metro Gate', 'Dwarka Sector 21',
    'Noida Sector 18', 'Gurgaon NH48', 'Cyber City Signal', 'AIIMS Flyover',
    'Rajiv Chowk', 'India Gate Circle', 'Lodhi Colony', 'Vasant Vihar',
  ];

  const cameraIds = [
    'CAM-001', 'CAM-002', 'CAM-003', 'CAM-004', 'CAM-005',
    'CAM-006', 'CAM-007', 'CAM-008', 'CAM-009', 'CAM-010',
    'CAM-011', 'CAM-012', 'CAM-013', 'CAM-014', 'CAM-015', 'CAM-016',
  ];

  const vehicleTypes = ['Car', 'Motorcycle', 'Truck', 'Bus', 'Auto', 'Van', 'SUV'];
  const statusOptions = ['Pending', 'Challan Issued', 'Paid', 'Under Review'];
  const statusColors = {
    'Pending': '#ff6b35',
    'Challan Issued': '#ffd700',
    'Paid': '#00ff88',
    'Under Review': '#00d4ff',
  };

  const officers = ['Insp. Sharma', 'Sgt. Verma', 'Cst. Patel', 'Insp. Singh', 'Sgt. Kumar'];

  const numberPlates = () => {
    const states = ['DL', 'MH', 'KA', 'TN', 'UP', 'HR', 'RJ', 'GJ'];
    const st = states[Math.floor(Math.random() * states.length)];
    const num = Math.floor(10 + Math.random() * 89);
    const letters = String.fromCharCode(65 + Math.floor(Math.random() * 26)) +
                    String.fromCharCode(65 + Math.floor(Math.random() * 26));
    const digits = Math.floor(1000 + Math.random() * 9000);
    return `${st} ${num} ${letters} ${digits}`;
  };

  const randomConfidence = () => (Math.random() * 15 + 85).toFixed(1);
  const randomSpeed = () => Math.floor(Math.random() * 80 + 30);
  const randomTime = () => {
    const h = String(Math.floor(Math.random() * 24)).padStart(2, '0');
    const m = String(Math.floor(Math.random() * 60)).padStart(2, '0');
    const s = String(Math.floor(Math.random() * 60)).padStart(2, '0');
    return `${h}:${m}:${s}`;
  };

  const generateViolation = (id) => {
    const vType = violationTypes[Math.floor(Math.random() * violationTypes.length)];
    const status = statusOptions[Math.floor(Math.random() * statusOptions.length)];
    return {
      id: id || `VIO-${String(Math.floor(Math.random() * 9000 + 1000)).padStart(6, '0')}`,
      vehicleNumber: numberPlates(),
      vehicleType: vehicleTypes[Math.floor(Math.random() * vehicleTypes.length)],
      violation: vType.name,
      violationIcon: vType.icon,
      violationColor: vType.color,
      fine: vType.fine,
      camera: cameraIds[Math.floor(Math.random() * cameraIds.length)],
      location: locations[Math.floor(Math.random() * locations.length)],
      time: randomTime(),
      date: '2026-08-03',
      confidence: randomConfidence(),
      speed: randomSpeed(),
      status: status,
      statusColor: statusColors[status],
      officer: officers[Math.floor(Math.random() * officers.length)],
      imageColor: `hsl(${Math.floor(Math.random() * 360)}, 60%, 40%)`,
    };
  };

  const generateCamera = (id) => {
    const isOnline = Math.random() > 0.15;
    return {
      id: cameraIds[id % cameraIds.length],
      location: locations[id % locations.length],
      status: isOnline ? 'Online' : 'Offline',
      resolution: ['1080p', '4K', '720p'][Math.floor(Math.random() * 3)],
      fps: [24, 30, 60][Math.floor(Math.random() * 3)],
      aiModel: ['YOLOv8', 'YOLOv9', 'RT-DETR'][Math.floor(Math.random() * 3)],
      lastMaintenance: '2026-07-15',
      detections: Math.floor(Math.random() * 500 + 100),
      uptime: (Math.random() * 10 + 90).toFixed(1) + '%',
      lat: 28.6 + (Math.random() - 0.5) * 0.3,
      lng: 77.2 + (Math.random() - 0.5) * 0.3,
    };
  };

  const generateUser = (id) => {
    const roles = ['Admin', 'Officer', 'Operator', 'Viewer'];
    const names = ['Rajesh Kumar', 'Priya Sharma', 'Amit Verma', 'Sunita Patel', 'Vikram Singh',
                   'Anjali Mehta', 'Rohit Gupta', 'Neha Joshi', 'Suresh Rao', 'Kavita Nair'];
    const role = roles[Math.floor(Math.random() * roles.length)];
    const roleColors = { Admin: '#ff3366', Officer: '#00d4ff', Operator: '#00ff88', Viewer: '#ffd700' };
    return {
      id: `USR-${String(id + 1).padStart(4, '0')}`,
      name: names[id % names.length],
      email: `user${id + 1}@trafficpd.gov.in`,
      role: role,
      roleColor: roleColors[role],
      status: Math.random() > 0.2 ? 'Active' : 'Inactive',
      lastLogin: '2026-08-03 09:30',
      permissions: Math.floor(Math.random() * 8 + 3),
    };
  };

  // Stats
  const stats = {
    activeCameras: 142,
    liveVehicles: 8734,
    totalViolations: 1247,
    numberPlateDetections: 34521,
    pendingChallans: 389,
    revenueCollected: 2847500,
    aiAccuracy: 97.3,
    systemHealth: 98.7,
  };

  // Hourly traffic data
  const hourlyTraffic = Array.from({ length: 24 }, (_, i) => ({
    hour: `${String(i).padStart(2, '0')}:00`,
    count: Math.floor(Math.sin((i - 6) * Math.PI / 12) * 3000 + 3500 + Math.random() * 500),
  }));

  // Daily traffic (7 days)
  const dailyTraffic = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map(day => ({
    day,
    vehicles: Math.floor(Math.random() * 20000 + 30000),
    violations: Math.floor(Math.random() * 500 + 800),
    revenue: Math.floor(Math.random() * 500000 + 200000),
  }));

  // Violation breakdown
  const violationBreakdown = violationTypes.map(v => ({
    name: v.name,
    count: Math.floor(Math.random() * 200 + 50),
    color: v.color,
  }));

  // Vehicle categories
  const vehicleCategories = vehicleTypes.map(v => ({
    name: v,
    count: Math.floor(Math.random() * 5000 + 1000),
  }));

  // Monthly revenue
  const monthlyRevenue = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].map(m => ({
    month: m,
    revenue: Math.floor(Math.random() * 1000000 + 1500000),
    challans: Math.floor(Math.random() * 2000 + 5000),
  }));

  return {
    violationTypes, locations, cameraIds, statusColors,
    generateViolation, generateCamera, generateUser,
    stats, hourlyTraffic, dailyTraffic, violationBreakdown, vehicleCategories, monthlyRevenue,
    numberPlates, randomConfidence, randomSpeed, randomTime,
  };
})();

window.TrafficData = TrafficData;
