// ============================================================
// Smart Traffic Crime Detection — Charts Module
// ============================================================

const Charts = (() => {
  const chartInstances = {};

  Chart.defaults.color = 'rgba(255,255,255,0.6)';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';
  Chart.defaults.font.family = "'Inter', sans-serif";

  const gradient = (ctx, color1, color2) => {
    const g = ctx.createLinearGradient(0, 0, 0, 400);
    g.addColorStop(0, color1);
    g.addColorStop(1, color2);
    return g;
  };

  const destroyChart = (id) => {
    if (chartInstances[id]) {
      chartInstances[id].destroy();
      delete chartInstances[id];
    }
  };

  // Hourly Traffic Line Chart
  const initHourlyTraffic = (canvasId) => {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const data = TrafficData.hourlyTraffic;

    chartInstances[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.map(d => d.hour),
        datasets: [{
          label: 'Vehicles',
          data: data.map(d => d.count),
          borderColor: '#00d4ff',
          backgroundColor: gradient(ctx, 'rgba(0,212,255,0.3)', 'rgba(0,212,255,0.01)'),
          fill: true,
          tension: 0.4,
          pointRadius: 3,
          pointBackgroundColor: '#00d4ff',
          pointHoverRadius: 6,
          borderWidth: 2,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(10,15,30,0.95)',
            borderColor: 'rgba(0,212,255,0.3)',
            borderWidth: 1,
            titleColor: '#00d4ff',
            bodyColor: 'rgba(255,255,255,0.8)',
            padding: 12,
          }
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { maxTicksLimit: 8 } },
          y: { grid: { color: 'rgba(255,255,255,0.04)' } }
        },
        animation: { duration: 800, easing: 'easeInOutQuart' },
      }
    });
  };

  // Daily Traffic Bar Chart
  const initDailyTraffic = (canvasId) => {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const data = TrafficData.dailyTraffic;

    chartInstances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.map(d => d.day),
        datasets: [
          {
            label: 'Vehicles',
            data: data.map(d => d.vehicles),
            backgroundColor: 'rgba(0,212,255,0.7)',
            borderRadius: 6,
            borderSkipped: false,
          },
          {
            label: 'Violations',
            data: data.map(d => d.violations),
            backgroundColor: 'rgba(255,51,102,0.7)',
            borderRadius: 6,
            borderSkipped: false,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            labels: { usePointStyle: true, pointStyle: 'circle', padding: 20 }
          },
          tooltip: {
            backgroundColor: 'rgba(10,15,30,0.95)',
            borderColor: 'rgba(0,212,255,0.3)',
            borderWidth: 1,
            titleColor: '#00d4ff',
            bodyColor: 'rgba(255,255,255,0.8)',
            padding: 12,
          }
        },
        scales: {
          x: { grid: { display: false } },
          y: { grid: { color: 'rgba(255,255,255,0.04)' } }
        },
        animation: { duration: 800 },
      }
    });
  };

  // Violation Donut Chart
  const initViolationDonut = (canvasId) => {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const top5 = TrafficData.violationBreakdown.slice(0, 6);

    chartInstances[canvasId] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: top5.map(d => d.name),
        datasets: [{
          data: top5.map(d => d.count),
          backgroundColor: ['#ff3366', '#ff6b35', '#ffd700', '#00d4ff', '#00ff88', '#9b59b6'],
          borderWidth: 0,
          hoverOffset: 8,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '70%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { usePointStyle: true, pointStyle: 'circle', padding: 12, font: { size: 10 } }
          },
          tooltip: {
            backgroundColor: 'rgba(10,15,30,0.95)',
            borderColor: 'rgba(0,212,255,0.3)',
            borderWidth: 1,
            titleColor: '#fff',
            bodyColor: 'rgba(255,255,255,0.8)',
            padding: 12,
          }
        },
        animation: { duration: 1000, animateRotate: true },
      }
    });
  };

  // Vehicle Category Pie Chart
  const initVehiclePie = (canvasId) => {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const data = TrafficData.vehicleCategories;

    chartInstances[canvasId] = new Chart(ctx, {
      type: 'pie',
      data: {
        labels: data.map(d => d.name),
        datasets: [{
          data: data.map(d => d.count),
          backgroundColor: ['#00d4ff', '#00ff88', '#ff6b35', '#ffd700', '#ff3366', '#9b59b6', '#1abc9c'],
          borderWidth: 0,
          hoverOffset: 8,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { usePointStyle: true, pointStyle: 'circle', padding: 10, font: { size: 10 } }
          },
          tooltip: {
            backgroundColor: 'rgba(10,15,30,0.95)',
            borderColor: 'rgba(0,212,255,0.3)',
            borderWidth: 1,
            titleColor: '#fff',
            bodyColor: 'rgba(255,255,255,0.8)',
            padding: 12,
          }
        },
        animation: { duration: 1000 },
      }
    });
  };

  // Revenue Area Chart
  const initRevenueChart = (canvasId) => {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const data = TrafficData.monthlyRevenue;

    chartInstances[canvasId] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.map(d => d.month),
        datasets: [
          {
            label: 'Revenue (₹)',
            data: data.map(d => d.revenue),
            borderColor: '#00ff88',
            backgroundColor: gradient(ctx, 'rgba(0,255,136,0.3)', 'rgba(0,255,136,0.01)'),
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointBackgroundColor: '#00ff88',
            pointHoverRadius: 7,
            borderWidth: 2,
            yAxisID: 'y',
          },
          {
            label: 'Challans',
            data: data.map(d => d.challans),
            borderColor: '#ff6b35',
            backgroundColor: gradient(ctx, 'rgba(255,107,53,0.3)', 'rgba(255,107,53,0.01)'),
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointBackgroundColor: '#ff6b35',
            pointHoverRadius: 7,
            borderWidth: 2,
            yAxisID: 'y1',
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            labels: { usePointStyle: true, pointStyle: 'circle', padding: 20 }
          },
          tooltip: {
            backgroundColor: 'rgba(10,15,30,0.95)',
            borderColor: 'rgba(0,212,255,0.3)',
            borderWidth: 1,
            titleColor: '#fff',
            bodyColor: 'rgba(255,255,255,0.8)',
            padding: 12,
          }
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' } },
          y: {
            type: 'linear', position: 'left',
            grid: { color: 'rgba(255,255,255,0.04)' },
            ticks: { callback: v => '₹' + (v / 100000).toFixed(1) + 'L' }
          },
          y1: {
            type: 'linear', position: 'right',
            grid: { drawOnChartArea: false },
          }
        },
        animation: { duration: 800 },
      }
    });
  };

  // AI Accuracy Gauge (using doughnut)
  const initAccuracyGauge = (canvasId, value) => {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    chartInstances[canvasId] = new Chart(ctx, {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [value, 100 - value],
          backgroundColor: ['#00ff88', 'rgba(255,255,255,0.05)'],
          borderWidth: 0,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '80%',
        rotation: -90,
        circumference: 180,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        animation: { duration: 1500, easing: 'easeInOutQuart' },
      }
    });
  };

  // Violation Trend (bar chart)
  const initViolationTrend = (canvasId) => {
    destroyChart(canvasId);
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const labels = Array.from({ length: 12 }, (_, i) => `${i + 1}:00`);
    const datasets = [
      { label: 'Overspeeding', color: '#ff3366' },
      { label: 'Helmet', color: '#ff6b35' },
      { label: 'Red Light', color: '#ffd700' },
    ].map(d => ({
      label: d.label,
      data: labels.map(() => Math.floor(Math.random() * 50 + 10)),
      backgroundColor: d.color + 'cc',
      borderRadius: 4,
      borderSkipped: false,
    }));

    chartInstances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, labels: { usePointStyle: true, pointStyle: 'circle', padding: 15 } },
          tooltip: {
            backgroundColor: 'rgba(10,15,30,0.95)',
            borderColor: 'rgba(0,212,255,0.3)',
            borderWidth: 1,
            padding: 12,
          }
        },
        scales: {
          x: { stacked: true, grid: { display: false } },
          y: { stacked: true, grid: { color: 'rgba(255,255,255,0.04)' } }
        },
        animation: { duration: 800 },
      }
    });
  };

  // Heatmap (custom canvas rendering)
  const initHeatmap = (canvasId) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.offsetWidth || 600;
    const H = canvas.offsetHeight || 200;
    canvas.width = W;
    canvas.height = H;
    const hours = 24;
    const days = 7;
    const cellW = W / hours;
    const cellH = H / days;
    const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

    for (let d = 0; d < days; d++) {
      for (let h = 0; h < hours; h++) {
        const intensity = Math.max(0, Math.sin((h - 7) * Math.PI / 8)) * Math.random();
        const r = Math.floor(intensity * 255);
        const g = Math.floor((1 - intensity) * 100 + intensity * 50);
        const b = Math.floor((1 - intensity) * 255);
        ctx.fillStyle = `rgba(${r},${g},${b},${0.3 + intensity * 0.7})`;
        ctx.fillRect(h * cellW + 1, d * cellH + 1, cellW - 2, cellH - 2);
      }
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.font = '10px Inter';
      ctx.fillText(dayNames[d], 4, d * cellH + cellH / 2 + 4);
    }
  };

  return { initHourlyTraffic, initDailyTraffic, initViolationDonut, initVehiclePie, initRevenueChart, initAccuracyGauge, initViolationTrend, initHeatmap, destroyChart };
})();

window.Charts = Charts;
