let clusterTelemetry = null;
let meshModels = [];
let lastMeshModelsRefresh = 0;
let graphPlan = null;
let activeTab = 'playground';
let activeModality = 'All';
let modelView = 'table';
let searchQuery = '';
let selectedChatModels = [];
let nodeModalTrigger = null;
let lastExecutionNodesRefresh = 0;

function formatBytesJS(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function getModelAccent(name) {
    if (!name) return '#58a6ff';
    if (name.includes('Llama')) return '#5b7cfa';
    if (name.includes('Qwen')) return '#835dff';
    if (name.includes('Gemma')) return '#7196ff';
    if (name.includes('Mistral')) return '#ff8e32';
    if (name.includes('Tiny')) return '#3fb950';
    if (name.includes('Spark')) return '#d29922';
    return '#58a6ff';
}

function switchTab(tabId) {
    activeTab = tabId;
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-view').forEach(view => {
        view.style.display = view.id === 'view-' + tabId ? 'block' : 'none';
    });
    if (tabId === 'models') {
        renderModelsList();
    }
}

function renderKernelWaterfall(events) {
    const canvas = document.getElementById('kernel-waterfall');
    const detail = document.getElementById('kernel-waterfall-detail');
    if (!canvas || !detail) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    if (!events || events.length === 0) {
        ctx.fillStyle = '#77817e';
        ctx.font = '13px ui-monospace, monospace';
        ctx.fillText('Waiting for recorded GPU dispatches...', 18, 32);
        detail.textContent = 'Waiting for recorded GPU dispatches.';
        return;
    }

    const positionEvents = events.filter(function(event) {
        return event.labels && event.labels.pos !== undefined;
    });
    let latestPrefill = -1;
    positionEvents.forEach(function(event, index) {
        if (event.labels.pos === '0') latestPrefill = index;
    });
    const generation = latestPrefill >= 0 ? positionEvents.slice(latestPrefill) : positionEvents;
    const prefill = generation.find(function(event) { return event.labels.pos === '0'; });
    const decodeEvents = generation.filter(function(event) {
        return event.labels.pos !== '0';
    });
    const steps = (decodeEvents.length ? decodeEvents : positionEvents.length ? positionEvents : events).slice(-28);
    const durations = steps.map(function(event) { return event.duration_us / 1000; });
    const total = durations.reduce(function(sum, duration) { return sum + duration; }, 0);
    const max = Math.max.apply(null, durations);
    const slowest = steps[durations.indexOf(max)];
    const plot = { left: 52, right: 16, top: 16, bottom: 28 };
    const plotWidth = width - plot.left - plot.right;
    const plotHeight = height - plot.top - plot.bottom;

    ctx.strokeStyle = '#1d2828';
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top);
    ctx.lineTo(plot.left, plot.top + plotHeight);
    ctx.lineTo(width - plot.right, plot.top + plotHeight);
    ctx.stroke();
    ctx.fillStyle = '#77817e';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(max.toFixed(1) + ' ms', 4, plot.top + 4);
    ctx.fillText('0 ms', 16, plot.top + plotHeight);

    const bars = [];
    const gap = 3;
    const barWidth = Math.max(3, (plotWidth - gap * (steps.length - 1)) / steps.length);
    steps.forEach(function(event, index) {
        const x = plot.left + index * (barWidth + gap);
        const h = Math.max(2, event.duration_us / 1000 / max * plotHeight);
        const y = plot.top + plotHeight - h;
        ctx.fillStyle = event === slowest ? '#d29922' : '#c4f000';
        ctx.fillRect(x, y, barWidth, h);
        bars.push({ event: event, x: x, width: barWidth });
        if (index === 0 || index === steps.length - 1 || index % 4 === 0) {
            const position = event.labels && event.labels.pos !== undefined ? event.labels.pos : index + 1;
            ctx.fillStyle = '#77817e';
            ctx.fillText(position, x, height - 8);
        }
    });
    const first = steps[0].labels && steps[0].labels.pos !== undefined ? steps[0].labels.pos : 1;
    const last = steps[steps.length - 1].labels && steps[steps.length - 1].labels.pos !== undefined ? steps[steps.length - 1].labels.pos : steps.length;
    const prefillText = prefill ? 'Prefill pos 0: ' + (prefill.duration_us / 1000).toFixed(2) + ' ms. ' : '';
    detail.textContent = prefillText + 'Decode pos ' + first + '–' + last + ': ' + (total / steps.length).toFixed(2) + ' ms avg · Peak: ' + max.toFixed(2) + ' ms. Click a bar for kernel dispatch details.';
    canvas.onclick = function(click) {
        const scale = canvas.width / canvas.getBoundingClientRect().width;
        const x = (click.clientX - canvas.getBoundingClientRect().left) * scale;
        const bar = bars.find(function(candidate) {
            return x >= candidate.x && x <= candidate.x + candidate.width;
        });
        if (!bar) return;
        const position = bar.event.labels && bar.event.labels.pos !== undefined ? 'position ' + bar.event.labels.pos : 'event';
        detail.textContent = position + ': ' + bar.event.name + ' ran for ' + (bar.event.duration_us / 1000).toFixed(3) + ' ms; grid ' + (bar.event.grid_size || []).join('×') + ', threadgroup ' + (bar.event.threadgroup_size || []).join('×') + '.';
    };
}

function rooflineLog(value) {
    return Math.log(value) / Math.LN10;
}

function rooflineValue(value) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

function renderRoofline(roofline) {
    const canvas = document.getElementById('roofline-canvas');
    const detail = document.getElementById('roofline-detail');
    if (!canvas || !detail) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const ceilings = (roofline && roofline.ceilings) || [];
    const points = (roofline && roofline.points) || [];
    const usable = ceilings.filter(function(ceiling) {
        return rooflineValue(ceiling.compute_tflops) && rooflineValue(ceiling.memory_bandwidth_gbs);
    });
    const observed = points.filter(function(point) {
        return point.state === 'observed' && rooflineValue(point.arithmetic_intensity) && rooflineValue(point.attained_tflops);
    });

    ctx.clearRect(0, 0, width, height);
    if (usable.length === 0) {
        ctx.fillStyle = '#77817e';
        ctx.font = '13px ui-monospace, monospace';
        ctx.fillText('No hardware capability ceilings are currently reported.', 18, 32);
        detail.textContent = 'No reported ceilings. Operational intensity and throughput remain unmeasured.';
        return;
    }

    const plot = { left: 66, right: 24, top: 22, bottom: 42 };
    const plotWidth = width - plot.left - plot.right;
    const plotHeight = height - plot.top - plot.bottom;
    const intensities = usable.map(function(ceiling) {
        return ceiling.compute_tflops * 1000 / ceiling.memory_bandwidth_gbs * 8;
    }).concat(observed.map(function(point) { return point.arithmetic_intensity; }), [0.125, 128]);
    const performances = usable.map(function(ceiling) {
        return ceiling.compute_tflops;
    }).concat(observed.map(function(point) { return point.attained_tflops; }), [0.01, 0.1]);
    const minIntensity = Math.min.apply(null, intensities) / 1.25;
    const maxIntensity = Math.max.apply(null, intensities) * 1.25;
    const minPerformance = Math.min.apply(null, performances) / 1.25;
    const maxPerformance = Math.max.apply(null, performances) * 1.25;
    const x = function(intensity) {
        return plot.left + (rooflineLog(intensity) - rooflineLog(minIntensity)) /
            (rooflineLog(maxIntensity) - rooflineLog(minIntensity)) * plotWidth;
    };
    const y = function(performance) {
        return plot.top + (rooflineLog(maxPerformance) - rooflineLog(performance)) /
            (rooflineLog(maxPerformance) - rooflineLog(minPerformance)) * plotHeight;
    };

    ctx.strokeStyle = '#1d2828';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top);
    ctx.lineTo(plot.left, plot.top + plotHeight);
    ctx.lineTo(plot.left + plotWidth, plot.top + plotHeight);
    ctx.stroke();
    ctx.fillStyle = '#77817e';
    ctx.font = '11px ui-monospace, monospace';
    [0.125, 1, 8, 64].forEach(function(intensity) {
        if (intensity >= minIntensity && intensity <= maxIntensity) {
            ctx.fillText(String(intensity), x(intensity) - 8, height - 20);
        }
    });
    [0.01, 0.1, 1, 10, 100].forEach(function(performance) {
        if (performance >= minPerformance && performance <= maxPerformance) {
            ctx.fillText(String(performance), 8, y(performance) + 4);
        }
    });
    ctx.fillText('Arithmetic intensity (FLOPs/byte, log)', plot.left + 330, height - 5);
    ctx.save();
    ctx.translate(14, plot.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Attained performance (TFLOPs, log)', 0, 0);
    ctx.restore();

    const colors = ['#58a6ff', '#d29922', '#3fb950', '#8957e5'];
    ctx.setLineDash([6, 4]);
    usable.forEach(function(ceiling, index) {
        const color = colors[index % colors.length];
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let step = 0; step <= 64; step++) {
            const intensity = minIntensity * Math.pow(maxIntensity / minIntensity, step / 64);
            const performance = Math.min(ceiling.compute_tflops, ceiling.memory_bandwidth_gbs * intensity / 1000);
            if (step === 0) {
                ctx.moveTo(x(intensity), y(Math.max(performance, minPerformance)));
            } else {
                ctx.lineTo(x(intensity), y(Math.max(performance, minPerformance)));
            }
        }
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.fillText(ceiling.node_id + ' heuristic ceiling', plot.left + 8, plot.top + 15 + index * 14);
    });
    ctx.setLineDash([]);

    observed.forEach(function(point) {
        ctx.fillStyle = '#c4f000';
        ctx.beginPath();
        ctx.arc(x(point.arithmetic_intensity), y(point.attained_tflops), 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillText(point.phase, x(point.arithmetic_intensity) + 8, y(point.attained_tflops) - 8);
    });

    const unmeasured = points.filter(function(point) { return point.state === 'unmeasured'; });
    const observedDetail = observed.map(function(point) {
        const flops = typeof point.flops === 'number' ? point.flops.toLocaleString() + ' FLOPs' : 'FLOPs unreported';
        const bytes = typeof point.bytes_moved === 'number' ? formatBytesJS(point.bytes_moved) + ' moved' : 'bytes unreported';
        return point.phase + ': observed ' + flops + ' · ' + bytes + ' · ' + point.attained_tflops.toFixed(3) + ' TFLOPs';
    });
    const unmeasuredDetail = unmeasured.map(function(point) {
        return point.phase + ' (' + (point.node_id || 'unassigned') + '): UNMEASURED — ' + point.reason;
    });
    detail.textContent = 'Dashed lines are heuristic capability ceilings, not observed throughput. ' +
        observedDetail.concat(unmeasuredDetail).join(' · ');
}

function meshRole(nodeID, assignment) {
    if (!assignment) return 'unassigned';
    const prefill = nodeID === assignment.prefill_node_id;
    const decode = nodeID === assignment.decode_node_id;
    if (prefill && decode) return 'prefill + decode';
    if (prefill) return 'prefill';
    if (decode) return 'decode';
    return 'worker';
}

function renderMeshTopology(mesh) {
    const canvas = document.getElementById('mesh-topology');
    const detail = document.getElementById('mesh-topology-detail');
    if (!canvas || !detail) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    if (!mesh || !mesh.self_node || !mesh.self_node.node_id) {
        ctx.fillStyle = '#77817e';
        ctx.font = '13px ui-monospace, monospace';
        ctx.fillText('No mesh node telemetry is available.', 18, 32);
        detail.textContent = 'No registered peers. Transport path mode is unavailable.';
        return;
    }

    const nodes = [mesh.self_node].concat(mesh.peers || []);
    const positions = new Map();
    const center = { x: width / 2, y: height / 2 };
    positions.set(mesh.self_node.node_id, center);
    const peers = nodes.slice(1);
    const radius = Math.min(width, height) * 0.34;
    peers.forEach(function(node, index) {
        const angle = -Math.PI / 2 + 2 * Math.PI * index / peers.length;
        positions.set(node.node_id, {
            x: center.x + radius * Math.cos(angle),
            y: center.y + radius * Math.sin(angle),
        });
    });

    ctx.strokeStyle = '#33403e';
    ctx.lineWidth = 1.5;
    peers.forEach(function(peer) {
        const position = positions.get(peer.node_id);
        ctx.beginPath();
        ctx.moveTo(center.x, center.y);
        ctx.lineTo(position.x, position.y);
        ctx.stroke();
    });

    const colors = {
        'prefill + decode': '#c4f000',
        prefill: '#58a6ff',
        decode: '#8957e5',
        worker: '#8b949e',
    };
    nodes.forEach(function(node) {
        const position = positions.get(node.node_id);
        const role = meshRole(node.node_id, mesh.active_assignment);
        const color = colors[role];
        ctx.fillStyle = '#0d1314';
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(position.x, position.y, 30, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#ebefee';
        ctx.font = '11px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.fillText((node.node_id || 'node').slice(0, 8), position.x, position.y - 2);
        ctx.fillStyle = color;
        ctx.fillText(role, position.x, position.y + 13);
    });
    ctx.textAlign = 'start';

    const endpointCount = nodes.reduce(function(total, node) {
        return total + ((node.endpoint_addrs || []).length);
    }, 0);
    const relayAdvertised = nodes.filter(function(node) {
        return (node.endpoint_addrs || []).some(function(address) {
            return address.indexOf('relay:') === 0;
        });
    }).length;
    const nonRelayAdvertised = nodes.filter(function(node) {
        return (node.endpoint_addrs || []).some(function(address) {
            return address.indexOf('relay:') !== 0;
        });
    }).length;
    const transportEvidence = [];
    if (relayAdvertised > 0) transportEvidence.push(relayAdvertised + ' node(s) advertise relay address(es)');
    if (nonRelayAdvertised > 0) transportEvidence.push(nonRelayAdvertised + ' node(s) advertise non-relay address(es)');
    detail.textContent = nodes.length + ' registered node(s); ' + endpointCount +
        ' advertised endpoint address(es). ' + (transportEvidence.join('; ') || 'No endpoint addresses are advertised.') +
        ' Lines show local peer registrations; active direct or relay path mode is not reported.';
    canvas.onclick = function(click) {
        const bounds = canvas.getBoundingClientRect();
        const scale = canvas.width / bounds.width;
        const x = (click.clientX - bounds.left) * scale;
        const y = (click.clientY - bounds.top) * scale;
        const selected = nodes.find(function(node) {
            const position = positions.get(node.node_id);
            return Math.hypot(x - position.x, y - position.y) <= 30;
        });
        if (selected) openNodeModal(selected.node_id);
    };
}

function meshText(value, suffix) {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value.toFixed(1) + suffix : 'unavailable';
}

function renderMeshSyncHUD(syncs) {
    const hud = document.getElementById('mesh-sync-hud');
    if (!hud) return;
    hud.replaceChildren();
    if (!syncs || syncs.length === 0) {
        const empty = document.createElement('div');
        empty.style.cssText = 'color:#77817e; font:12px ui-monospace, monospace; padding:6px 0;';
        empty.textContent = 'No hfsync jobs have been reported.';
        hud.append(empty);
        return;
    }

    syncs.slice().reverse().forEach(function(sync) {
        const card = document.createElement('div');
        card.style.cssText = 'border:1px solid #1d2828; border-radius:6px; background:#0b1112; padding:12px;';
        const title = document.createElement('div');
        title.style.cssText = 'color:#ebefee; font:600 12px ui-monospace, monospace;';
        title.textContent = sync.model_id || 'model sync';
        const meta = document.createElement('div');
        meta.style.cssText = 'color:#77817e; font:11px ui-monospace, monospace; margin-top:6px;';
        const progress = sync.total_bytes > 0 ? (sync.percent || 0).toFixed(1) + '%' : 'progress unavailable';
        const file = sync.current_file ? ' · file ' + sync.current_file : '';
        const error = sync.error ? ' · error ' + sync.error : '';
        meta.textContent = (sync.state || 'unknown') + ' · ' + progress + ' · ' +
            meshText(sync.throughput_mbs, ' MB/s') + file + error;
        card.append(title, meta);
        hud.append(card);
    });
}

function pgoNumber(value, unit) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) + unit : 'unmeasured';
}

function pgoEmpty(container, message) {
    container.replaceChildren();
    const item = document.createElement('div');
    item.style.cssText = 'color:#77817e; font:11px ui-monospace, monospace;';
    item.textContent = message;
    container.append(item);
}

function renderPGO(response) {
    const canvas = document.getElementById('pgo-trials');
    const detail = document.getElementById('pgo-detail');
    const state = document.getElementById('pgo-state');
    const history = document.getElementById('pgo-history');
    const hotSwaps = document.getElementById('pgo-hotswaps');
    if (!canvas || !detail || !state || !history || !hotSwaps) return;

    const pgo = response && response.data;
    const observed = response && response.state === 'observed' && pgo;
    state.textContent = observed ? 'observed' : (response && response.state) || 'unavailable';
    state.style.background = observed ? '#1f6feb' : '#30363d';
    state.style.color = observed ? '#ffffff' : '#a9b2af';

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    if (!observed) {
        ctx.fillStyle = '#77817e';
        ctx.font = '13px ui-monospace, monospace';
        ctx.fillText('PGO is inactive: no observed auto-tuning trial data.', 18, 32);
        detail.textContent = ((response && response.reason) || 'PGO telemetry is unavailable.') +
            ' FlightRecorder dispatches: ' + ((response && response.flight_recorder_dispatches) || 0) + '.';
        pgoEmpty(history, 'No optimization iterations have been observed.');
        pgoEmpty(hotSwaps, 'No pipeline hot swaps have been observed.');
        return;
    }

    const trials = pgo.autotuning_trials || [];
    const plot = { left: 52, right: 16, top: 16, bottom: 28 };
    const plotWidth = width - plot.left - plot.right;
    const plotHeight = height - plot.top - plot.bottom;
    ctx.strokeStyle = '#1d2828';
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top);
    ctx.lineTo(plot.left, plot.top + plotHeight);
    ctx.lineTo(width - plot.right, plot.top + plotHeight);
    ctx.stroke();

    if (trials.length === 0) {
        ctx.fillStyle = '#77817e';
        ctx.font = '13px ui-monospace, monospace';
        ctx.fillText('No observed auto-tuning trials.', 18, 32);
    } else {
        const maxDuration = Math.max.apply(null, trials.map(function(trial) { return trial.duration_us; }));
        const minDuration = Math.min.apply(null, trials.map(function(trial) { return trial.duration_us; }));
        const durationRange = Math.max(1, maxDuration - minDuration);
        trials.forEach(function(trial, index) {
            const x = plot.left + (index + 0.5) * plotWidth / trials.length;
            const y = plot.top + (maxDuration - trial.duration_us) / durationRange * plotHeight;
            ctx.fillStyle = '#c4f000';
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
            ctx.fillStyle = '#77817e';
            ctx.font = '10px ui-monospace, monospace';
            ctx.fillText((trial.threadgroup_size || []).join('×'), x - 12, height - 8);
        });
        ctx.fillStyle = '#77817e';
        ctx.font = '11px ui-monospace, monospace';
        ctx.fillText(maxDuration.toFixed(2) + ' µs', 4, plot.top + 4);
        ctx.fillText(minDuration.toFixed(2) + ' µs', 4, plot.top + plotHeight);
    }

    const convergence = pgo.convergence || {};
    detail.textContent = 'Target: ' + (pgo.active_target || 'unreported') + ' · ' +
        trials.length + ' observed trial(s) · convergence ' + (convergence.locked ? 'locked' : 'not locked') +
        ' · confidence ' + pgoNumber(convergence.confidence_score, '');

    history.replaceChildren();
    const iterations = pgo.history || [];
    if (iterations.length === 0) {
        pgoEmpty(history, 'No optimization iterations have been observed.');
    } else {
        iterations.forEach(function(iteration) {
            const item = document.createElement('div');
            item.style.cssText = 'color:#a9b2af; font:11px ui-monospace, monospace;';
            item.textContent = 'Iteration ' + iteration.index + ': ' + iteration.name + ' · ' +
                pgoNumber(iteration.latency_ms, ' ms') + ' · ' + pgoNumber(iteration.speedup_pct, '%') +
                (iteration.active ? ' · active' : '');
            history.append(item);
        });
    }

    hotSwaps.replaceChildren();
    const swaps = pgo.hot_swaps || [];
    if (swaps.length === 0) {
        pgoEmpty(hotSwaps, 'No pipeline hot swaps have been observed.');
    } else {
        swaps.forEach(function(swap) {
            const item = document.createElement('div');
            item.style.cssText = 'color:#a9b2af; font:11px ui-monospace, monospace;';
            item.textContent = 'Hot swap: ' + swap.kernel_name + ' · ' + swap.from_desc + ' → ' + swap.to_desc +
                (swap.reason ? ' · ' + swap.reason : '');
            hotSwaps.append(item);
        });
    }
}

async function refreshPGO() {
    try {
        const res = await fetch('/api/optimizer/pgo');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        renderPGO(await res.json());
    } catch (err) {
        renderPGO({ state: 'unavailable', reason: 'PGO endpoint unavailable: ' + err.message });
    }
}

function memoryCard(title, state, detail) {
    const card = document.createElement('div');
    card.style.cssText = 'border:1px solid #1d2828; border-radius:6px; background:#0b1112; padding:12px;';
    const header = document.createElement('div');
    header.style.cssText = 'display:flex; justify-content:space-between; gap:12px; align-items:baseline;';
    const name = document.createElement('strong');
    name.style.cssText = 'color:#ebefee; font:600 12px ui-monospace, monospace;';
    name.textContent = title;
    const badge = document.createElement('span');
    const observed = state === 'observed';
    const configured = state === 'configured';
    badge.style.cssText = 'font:600 10px ui-monospace, monospace; color:' +
        (observed ? '#ffffff' : configured ? '#c4f000' : '#a9b2af') + ';';
    badge.textContent = (state || 'unavailable').toUpperCase();
    header.append(name, badge);
    const body = document.createElement('div');
    body.style.cssText = 'color:#77817e; font:11px ui-monospace, monospace; margin-top:6px; line-height:1.45;';
    body.textContent = detail;
    card.append(header, body);
    return card;
}

function renderMemoryTelemetry(response) {
    const kv = document.getElementById('memory-kv');
    const prefix = document.getElementById('memory-prefix');
    const precision = document.getElementById('memory-precision');
    const speculation = document.getElementById('memory-speculation');
    const batching = document.getElementById('memory-batching');
    if (!kv || !prefix || !precision || !speculation || !batching) return;

    const memory = response || {};
    const kvCache = memory.kv_cache || {};
    const configured = kvCache.state === 'configured';
    const observedKV = kvCache.state === 'observed';
    const kvDetail = observedKV ?
        formatBytesJS(kvCache.allocated_bytes || 0) + ' allocated · ' +
        (kvCache.active_tokens || 0) + ' active / ' + kvCache.max_sequence_tokens +
        ' tokens · ' + kvCache.layers + ' layers. Fragmentation and block counters are unavailable.' :
        configured ?
            'Configured for ' + kvCache.max_sequence_tokens + ' tokens · ' + kvCache.layers +
            ' layers · ' + kvCache.kv_heads + ' KV heads · head dimension ' + kvCache.head_dim +
            '. Allocation, occupancy, fragmentation, and block counters are unavailable.' :
            (kvCache.reason || 'KV-cache capacity is unavailable.');
    kv.replaceChildren(memoryCard('KV cache', kvCache.state, kvDetail));

    const prefixCache = memory.prefix_cache || {};
    const prefixDetail = prefixCache.state === 'observed' ?
        (prefixCache.entries || 0) + ' snapshot(s) · ' + formatBytesJS(prefixCache.bytes || 0) +
        ' · ' + (prefixCache.hits || 0) + ' hit(s) / ' + (prefixCache.misses || 0) +
        ' miss(es) · ' + (prefixCache.saved_tokens || 0) + ' tokens skipped.' :
        (prefixCache.reason || 'Prefix-cache telemetry is unavailable.');
    prefix.replaceChildren(memoryCard('Prefix cache', prefixCache.state, prefixDetail));

    const precisionData = memory.precision || {};
    const dtypes = precisionData.dtypes || [];
    const precisionCard = memoryCard('Tensor precision', precisionData.state,
        precisionData.state === 'observed' ? 'Observed safetensors dtype metadata from the active engine.' :
            (precisionData.reason || 'Tensor precision metadata is unavailable.'));
    if (precisionData.state === 'observed' && dtypes.length > 0) {
        const rows = document.createElement('div');
        rows.style.cssText = 'display:grid; gap:4px; margin-top:8px; color:#a9b2af; font:11px ui-monospace, monospace;';
        dtypes.forEach(function(dtype) {
            const row = document.createElement('div');
            row.textContent = dtype.dtype + ' · ' + dtype.tensor_count + ' tensor(s) · ' + formatBytesJS(dtype.bytes);
            rows.append(row);
        });
        precisionCard.append(rows);
    }
    precision.replaceChildren(precisionCard);

    const speculationData = memory.speculation || {};
    speculation.replaceChildren(memoryCard('Speculative decoding', speculationData.state,
        speculationData.reason || 'Speculative-decoding telemetry is unavailable.'));

    const batchingData = memory.batching || {};
    batching.replaceChildren(memoryCard('Batching', batchingData.state,
        batchingData.reason || 'Batching telemetry is unavailable.'));
}

async function refreshMemoryTelemetry() {
    try {
        const res = await fetch('/api/memory/kvcache');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        renderMemoryTelemetry(await res.json());
    } catch (err) {
        const reason = 'Memory telemetry endpoint unavailable: ' + err.message;
        renderMemoryTelemetry({
            kv_cache: { state: 'unavailable', reason: reason },
            prefix_cache: { state: 'unavailable', reason: reason },
            precision: { state: 'unavailable', reason: reason },
            speculation: { state: 'unavailable', reason: reason },
            batching: { state: 'unavailable', reason: reason }
        });
    }
}

function requestSlotDuration(value) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) + ' ms' : 'in progress';
}

function requestSlotLine(slot) {
    const line = document.createElement('div');
    line.style.cssText = 'border:1px solid #1d2828; border-radius:6px; background:#0b1112; padding:10px 12px; color:#a9b2af; font:11px ui-monospace, monospace;';
    const title = document.createElement('strong');
    title.style.cssText = 'color:#ebefee; font:600 11px ui-monospace, monospace;';
    title.textContent = (slot.state || 'unknown').toUpperCase() + ' · ' + (slot.kind || 'request') + ' · ' + (slot.stream ? 'stream' : 'response');
    const detail = document.createElement('div');
    detail.style.cssText = 'margin-top:4px;';
    const user = slot.user ? 'client user: ' + slot.user + ' (unauthenticated)' : 'client user: unavailable';
    const position = slot.state === 'queued' && slot.queue_position ? ' · queue position: ' + slot.queue_position : '';
    const queue = typeof slot.queue_duration_ms === 'number' ? ' · queue time: ' + requestSlotDuration(slot.queue_duration_ms) : '';
    const runtime = typeof slot.runtime_duration_ms === 'number' ? ' · runtime: ' + requestSlotDuration(slot.runtime_duration_ms) : '';
    detail.textContent = (slot.id || 'request') + position + queue + runtime + ' · ' + user;
    line.append(title, detail);
    return line;
}

function renderRequestSlotQueue(queue) {
    const container = document.getElementById('request-slot-queue');
    if (!container) return;
    container.replaceChildren();
    if (!queue || queue.state !== 'observed') {
        const unavailable = document.createElement('div');
        unavailable.style.cssText = 'color:#77817e; font:12px ui-monospace, monospace;';
        unavailable.textContent = (queue && queue.reason) || 'Request-slot scheduler telemetry is unavailable.';
        container.append(unavailable);
        return;
    }

    const summary = document.createElement('div');
    summary.style.cssText = 'color:#ebefee; font:600 12px ui-monospace, monospace;';
    summary.textContent = 'Running: ' + queue.running + ' request(s) · Waiting: ' + queue.queued +
        ' request(s) · Capacity: ' + queue.capacity + ' request.';
    container.append(summary);
    const policy = document.createElement('div');
    policy.style.cssText = 'color:#77817e; font:11px ui-monospace, monospace;';
    policy.textContent = queue.reason;
    container.append(policy);
    const slots = queue.slots || [];
    if (slots.length === 0) {
        const empty = document.createElement('div');
        empty.style.cssText = 'color:#77817e; font:11px ui-monospace, monospace;';
        empty.textContent = 'No request currently holds or waits for an inference slot.';
        container.append(empty);
    } else {
        slots.forEach(function(slot) { container.append(requestSlotLine(slot)); });
    }
    const recent = queue.recent || [];
    if (recent.length > 0) {
        const recentTitle = document.createElement('div');
        recentTitle.style.cssText = 'color:#77817e; font:600 10px ui-monospace, monospace; margin-top:4px; text-transform:uppercase;';
        recentTitle.textContent = 'Latest completed or canceled requests';
        container.append(recentTitle);
        recent.slice().reverse().slice(0, 3).forEach(function(slot) { container.append(requestSlotLine(slot)); });
    }
}

function renderSpeculativeTree(speculation) {
    const canvas = document.getElementById('speculative-tree-canvas');
    const detail = document.getElementById('speculative-tree-detail');
    if (!canvas || !detail) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const trace = speculation && speculation.trace;
    if (!speculation || speculation.state !== 'observed' || !trace) {
        ctx.fillStyle = '#77817e';
        ctx.font = '13px ui-monospace, monospace';
        ctx.fillText('No paired draft/target tree is currently reported.', 18, 32);
        detail.textContent = (speculation && speculation.reason) || 'Speculative decoding telemetry is unavailable.';
        return;
    }
    const nodes = trace.nodes || [];
    if (nodes.length === 0) {
        ctx.fillStyle = '#77817e';
        ctx.font = '13px ui-monospace, monospace';
        ctx.fillText('The paired comparison reported no renderable nodes.', 18, 32);
        detail.textContent = speculation.reason || 'Paired execution reported no renderable tree nodes.';
        return;
    }

    const byID = new Map(nodes.map(function(node) { return [node.id, node]; }));
    const depth = new Map();
    function nodeDepth(node) {
        if (depth.has(node.id)) return depth.get(node.id);
        const seen = new Set([node.id]);
        let value = 0;
        let parent = node.parent;
        while (parent && byID.has(parent) && !seen.has(parent)) {
            seen.add(parent);
            value++;
            parent = byID.get(parent).parent;
        }
        depth.set(node.id, value);
        return value;
    }
    const levels = new Map();
    nodes.forEach(function(node) {
        const level = nodeDepth(node);
        if (!levels.has(level)) levels.set(level, []);
        levels.get(level).push(node);
    });
    const maxDepth = Math.max.apply(null, Array.from(levels.keys()));
    const positions = new Map();
    levels.forEach(function(levelNodes, level) {
        levelNodes.forEach(function(node, index) {
            positions.set(node.id, {
                x: 60 + (canvas.width - 120) * level / Math.max(1, maxDepth),
                y: 46 + (canvas.height - 92) * (index + 1) / (levelNodes.length + 1),
            });
        });
    });

    ctx.strokeStyle = '#33403e';
    ctx.lineWidth = 1.5;
    nodes.forEach(function(node) {
        const parent = positions.get(node.parent);
        const current = positions.get(node.id);
        if (!parent || !current) return;
        ctx.beginPath();
        ctx.moveTo(parent.x, parent.y);
        ctx.lineTo(current.x, current.y);
        ctx.stroke();
    });
    const colors = { accepted: '#3fb950', rejected: '#d29922', target: '#58a6ff' };
    nodes.forEach(function(node) {
        const position = positions.get(node.id);
        ctx.fillStyle = colors[node.state] || '#77817e';
        ctx.beginPath();
        ctx.arc(position.x, position.y, 16, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = '#080c0d';
        ctx.font = '10px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(String(node.token), position.x, position.y + 3);
        ctx.fillStyle = '#a9b2af';
        ctx.font = '10px ui-monospace, monospace';
        ctx.fillText((node.engine || 'engine') + ' · ' + (node.state || 'state'), position.x, position.y + 31);
    });
    ctx.textAlign = 'start';
    detail.textContent = 'Draft node: ' + (trace.draft_node_id || 'unreported') + ' · target node: ' +
        (trace.target_node_id || 'unreported') + ' · draft tokens: ' + (trace.draft || []).length +
        ' · target tokens: ' + (trace.target || []).length + ' · accepted: ' + trace.accepted + '. ' +
        (speculation.reason || 'Paired comparison only; OpenAI generation is not accelerated by this trace.');
}

async function refreshRuntimeScheduling() {
    try {
        const res = await fetch('/api/runtime/scheduling');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        renderRequestSlotQueue(data.request_queue);
        renderSpeculativeTree(data.speculation);
    } catch (err) {
        const reason = 'Runtime scheduling endpoint unavailable: ' + err.message;
        renderRequestSlotQueue({ state: 'unavailable', reason: reason });
        renderSpeculativeTree({ state: 'unavailable', reason: reason });
    }
}

function graphNodeElement(node) {
    const element = document.createElement('div');
    const colors = { apple_ane: '#3fb950', metal_gpu: '#58a6ff', host_cpu: '#8b949e', remote_tensor: '#d29922' };
    element.style.cssText = 'border:1px solid ' + (colors[node.target] || '#1d2828') + '; border-radius:6px; padding:8px; min-width:86px; background:#0b1112;';
    const op = document.createElement('div');
    op.textContent = node.op;
    op.style.cssText = 'font:600 12px ui-monospace, monospace; color:#ebefee;';
    const target = document.createElement('div');
    target.textContent = node.target;
    target.style.cssText = 'font-size:10px; color:' + (colors[node.target] || '#8b949e') + '; margin-top:4px; text-transform:uppercase;';
    element.append(op, target);
    if (node.children && node.children.length) {
        const children = document.createElement('div');
        children.style.cssText = 'display:flex; gap:6px; margin-top:8px; padding-left:8px; border-left:1px solid #1d2828;';
        node.children.forEach(function(child) { children.append(graphNodeElement(child)); });
        element.append(children);
    }
    return element;
}

function showGraphPlan(which) {
    const container = document.getElementById('graph-plan');
    if (!container || !graphPlan) return;
    container.replaceChildren(graphNodeElement(graphPlan[which]));
}

async function refreshGraphPlan() {
    try {
        const res = await fetch('/api/graph/plan');
        graphPlan = await res.json();
        showGraphPlan(graphPlan.fused ? 'after' : 'before');
        const rewrites = (graphPlan.rewrites || []).map(function(rewrite) {
            return rewrite.name + ': ' + (rewrite.matched ? 'matched' : 'not matched');
        });
        const targetEl = document.getElementById('graph-rewrites');
        if (targetEl) {
            targetEl.textContent = rewrites.join(' · ') + ' · ' + graphPlan.saturations + ' recorded saturation iterations';
        }
    } catch (err) {
        const targetEl = document.getElementById('graph-rewrites');
        if (targetEl) targetEl.textContent = 'Graph plan unavailable: ' + err.message;
    }
}

async function refreshTelemetry() {
    try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        clusterTelemetry = data;

        if (Date.now() - lastMeshModelsRefresh >= 5000) {
            lastMeshModelsRefresh = Date.now();
            refreshModelSelector();
        }
        if (Date.now() - lastExecutionNodesRefresh >= 5000) {
            lastExecutionNodesRefresh = Date.now();
            refreshExecutionTargets();
        }

        const dispatchesEl = document.getElementById('dispatches');
        if (dispatchesEl) dispatchesEl.innerText = data.total_dispatches;
        const p50El = document.getElementById('p50');
        if (p50El) p50El.innerText = data.p50_latency_us.toFixed(1) + ' µs';
        const p95El = document.getElementById('p95');
        if (p95El) p95El.innerText = data.p95_latency_us.toFixed(1) + ' µs';
        const toksEl = document.getElementById('toks');
        if (toksEl) toksEl.innerText = data.effective_tok_s.toFixed(1) + ' tok/s';
        const planEl = document.getElementById('plan');
        if (planEl) planEl.innerText = data.active_plan;

        renderKernelWaterfall(data.recent_timeline || []);
        renderRoofline(data.roofline);
        renderMeshTopology(data.cluster_mesh);
        renderMeshSyncHUD(data.active_syncs || []);
        refreshPGO();
        refreshMemoryTelemetry();
        refreshRuntimeScheduling();

        // Sync status banner update
        const banner = document.getElementById('sync-banner');
        if (banner) {
            if (data.active_syncs && data.active_syncs.length > 0) {
                const latest = data.active_syncs[data.active_syncs.length - 1];
                banner.style.display = 'block';
                document.getElementById('sync-model-id').innerText = latest.model_id;
                document.getElementById('sync-peer').innerText = (latest.source_platform || 'Peer') + ' (' + (latest.source_node_id ? latest.source_node_id.slice(0, 8) : 'cluster') + ')';
                document.getElementById('sync-speed').innerText = meshText(latest.throughput_mbs, ' MB/s');
                document.getElementById('sync-pct').innerText = latest.total_bytes > 0 ? (latest.percent || 0).toFixed(1) + '%' : 'progress unavailable';
                document.getElementById('sync-progress').style.width = latest.total_bytes > 0 ? Math.max(0, Math.min(100, latest.percent || 0)) + '%' : '0%';
                document.getElementById('sync-file').innerText = latest.current_file ? 'File: ' + latest.current_file : (latest.state === 'completed' ? 'Synchronization Complete!' : '');
                if (latest.state === 'completed') {
                    document.getElementById('sync-speed').innerText = latest.throughput_mbs > 0 ? 'Done (' + latest.throughput_mbs.toFixed(1) + ' MB/s avg)' : 'Done';
                }
            } else {
                banner.style.display = 'none';
            }
        }

        if (data.recent_generations && data.recent_generations.length > 0) {
            const tbody = document.getElementById('gen-table');
            if (tbody) {
                tbody.innerHTML = data.recent_generations.slice().reverse().map(function(g) {
                    const pShort = g.prefill_node ? g.prefill_node.slice(0, 8) : '';
                    const dShort = g.decode_node ? g.decode_node.slice(0, 8) : '';
                    const timeStr = new Date(g.timestamp).toLocaleTimeString();
                    return '<tr class="model-row">' +
                        '<td style="color:#77817e;">' + timeStr + '</td>' +
                        '<td style="color:#ebefee; max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="' + g.prompt + '">' + g.prompt + '</td>' +
                        '<td><span class="badge" style="background:#1f6feb; color:white;">' + g.prefill_platform + '</span> <code>' + pShort + '</code></td>' +
                        '<td><span class="badge" style="background:#8957e5; color:white;">' + g.decode_platform + '</span> <code>' + dShort + '</code></td>' +
                        '<td style="color:#c4f000; font-weight:600;">' + g.tokens_generated + ' tok (' + g.tok_per_sec.toFixed(1) + ' tok/s)</td>' +
                        '<td style="color:#77817e;">' + g.duration_ms.toFixed(1) + ' ms</td>' +
                    '</tr>';
                }).join('');
            }
        }
    } catch (e) {}
}

refreshTelemetry();
setInterval(refreshTelemetry, 1000);
refreshGraphPlan();

async function refreshModelSelector() {
    const select = document.getElementById('mesh-model-select');
    const button = document.getElementById('mesh-model-sync');
    const selfID = clusterTelemetry && clusterTelemetry.cluster_mesh && clusterTelemetry.cluster_mesh.self_node.node_id;
    
    try {
        const res = await fetch('/api/mesh/models');
        const models = await res.json();
        meshModels = Array.isArray(models) ? models : [];
        renderModelsList();

        if (select && selfID) {
            const wanted = select.value;
            const remoteModels = meshModels.filter(function(model) {
                return (model.nodes || []).some(function(node) {
                    return node.node_id !== selfID;
                });
            });

            select.replaceChildren();
            if (remoteModels.length === 0) {
                const option = document.createElement('option');
                option.textContent = 'No remote model snapshots available';
                select.append(option);
                select.disabled = true;
                if (button) button.disabled = true;
                return;
            }

            remoteModels.forEach(function(model) {
                const source = model.nodes.find(function(node) {
                    return node.node_id !== selfID;
                });
                const option = document.createElement('option');
                option.value = model.model_id;
                option.textContent = model.model_id + ' (' + formatBytesJS(model.total_bytes) + ' from ' + source.os.toUpperCase() + '/' + source.arch + ')';
                select.append(option);
            });
            if (remoteModels.some(function(model) { return model.model_id === wanted; })) {
                select.value = wanted;
            }
            select.disabled = false;
            if (button) button.disabled = false;
        }
    } catch (err) {
        if (select) select.disabled = true;
        if (button) button.disabled = true;
    }
}

function filterModality(modality) {
    activeModality = modality;
    document.querySelectorAll('.modality-tabs button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.modality === modality);
    });
    renderModelsList();
}

function setModelViewMode(mode) {
    modelView = mode;
    document.querySelectorAll('.view-switch button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === mode);
    });
    renderModelsList();
}

function renderModelsList() {
    const tableContainer = document.getElementById('models-table-view');
    const cardsContainer = document.getElementById('models-cards-view');
    const countEl = document.getElementById('models-count-summary');
    if (!tableContainer || !cardsContainer) return;

    let filtered = meshModels.filter(m => {
        if (searchQuery && !m.model_id.toLowerCase().includes(searchQuery.toLowerCase())) {
            return false;
        }
        if (activeModality !== 'All') {
            if (activeModality === 'NVFP4' && !m.model_id.toUpperCase().includes('NVFP4')) return false;
            if (activeModality === 'MXFP4' && !m.model_id.toUpperCase().includes('MXFP4')) return false;
            if (activeModality === 'FP8' && !m.model_id.toUpperCase().includes('FP8')) return false;
            if (activeModality === 'BF16' && !m.model_id.toLowerCase().includes('bf16')) return false;
            if (activeModality === 'GGUF' && !m.model_id.toUpperCase().includes('GGUF')) return false;
        }
        return true;
    });

    if (countEl) countEl.innerText = `${filtered.length} models discovered across P2P cluster mesh`;

    if (modelView === 'cards') {
        tableContainer.style.display = 'none';
        cardsContainer.style.display = 'grid';
        cardsContainer.innerHTML = filtered.map(m => {
            const initial = m.model_id.split('/').pop().charAt(0).toUpperCase();
            const accent = getModelAccent(m.model_id);
            const nodesHtml = (m.nodes || []).map(n => `<span class="badge" style="background:#111616; border:1px solid #1d2828; color:#a9b2af; font-size:11px; margin-right:4px;"><b>${n.os.toUpperCase()}</b> <code>${n.node_id.slice(0,6)}</code></span>`).join('');
            return `<div class="model-card">
                <div>
                    <span class="model-mark" style="--model-color: ${accent}">${initial}</span>
                    <span>${m.model_id.split('/')[0]}</span>
                    <strong>${m.model_id.split('/').pop()}</strong>
                </div>
                <dl>
                    <div><dt>Disk Size</dt><dd>${formatBytesJS(m.total_bytes)}</dd></div>
                    <div><dt>Files</dt><dd>${m.file_count}</dd></div>
                    <div style="grid-column: span 2;"><dt>Hosted on</dt><dd style="margin-top:6px;">${nodesHtml}</dd></div>
                </dl>
                <div style="margin-top:14px; display:flex; gap:8px;">
                    <button class="button" style="flex:1;" onclick="openModelDrawer('${m.model_id}')">Details</button>
                    <button class="primary" style="flex:1;" onclick="useModelInChat('${m.model_id}')">Use Model</button>
                </div>
            </div>`;
        }).join('');
    } else {
        cardsContainer.style.display = 'none';
        tableContainer.style.display = 'block';
        const tbody = document.getElementById('models-tbody');
        if (tbody) {
            tbody.innerHTML = filtered.map(m => {
                const initial = m.model_id.split('/').pop().charAt(0).toUpperCase();
                const accent = getModelAccent(m.model_id);
                const nodesHtml = (m.nodes || []).map(n => `<span class="badge" style="background:#111616; border:1px solid #1d2828; color:#a9b2af; font-size:11px; margin-right:4px;"><b>${n.os.toUpperCase()}</b> <code>${n.node_id.slice(0,6)}</code></span>`).join('');
                return `<div class="model-row" onclick="openModelDrawer('${m.model_id}')">
                    <span class="model-name">
                        <span class="model-mark" style="--model-color: ${accent}">${initial}</span>
                        <span>${m.model_id}</span>
                    </span>
                    <span>${formatBytesJS(m.total_bytes)}</span>
                    <span>${m.file_count} files</span>
                    <span style="grid-column: span 3; text-align: left;">${nodesHtml}</span>
                    <span style="display:flex; justify-content:flex-end; gap:6px;">
                        <button class="button" style="padding:3px 8px;" onclick="event.stopPropagation(); useModelInChat('${m.model_id}')">Chat</button>
                    </span>
                </div>`;
            }).join('');
        }
    }
}

function openModelDrawer(modelId) {
    const model = meshModels.find(m => m.model_id === modelId);
    if (!model) return;
    const drawer = document.getElementById('model-drawer');
    if (!drawer) return;
    
    document.getElementById('drawer-model-name').innerText = model.model_id;
    document.getElementById('drawer-model-size').innerText = formatBytesJS(model.total_bytes);
    document.getElementById('drawer-model-files').innerText = model.file_count + ' snapshot files';
    
    const nodesContainer = document.getElementById('drawer-model-nodes');
    nodesContainer.innerHTML = (model.nodes || []).map(n => {
        return `<div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1d2828;">
            <div>
                <b style="color:#ebefee;">${n.os.toUpperCase()} / ${n.arch}</b>
                <div style="font-size:11px; color:#77817e;"><code>${n.node_id}</code></div>
            </div>
            <div style="text-align:right;">
                <span style="color:#c4f000; font-weight:600;">${n.tflops} TFLOPs</span> · <span style="color:#58a6ff;">${n.bandwidth} GB/s</span>
            </div>
        </div>`;
    }).join('');

    drawer.style.display = 'block';
}

function closeModelDrawer() {
    const drawer = document.getElementById('model-drawer');
    if (drawer) drawer.style.display = 'none';
}

function useModelInChat(modelId) {
    closeModelDrawer();
    switchTab('playground');
    const input = document.getElementById('prompt-input');
    if (input) input.focus();
}

function syncSelectedMeshModel() {
    const select = document.getElementById('mesh-model-select');
    const selfID = clusterTelemetry && clusterTelemetry.cluster_mesh && clusterTelemetry.cluster_mesh.self_node.node_id;
    const model = meshModels.find(function(candidate) {
        return candidate.model_id === select.value;
    });
    const source = model && model.nodes.find(function(node) {
        return node.node_id !== selfID;
    });
    if (source) triggerSync(model.model_id, source.node_id);
}

async function triggerSync(modelId, sourceNodeId) {
    try {
        const banner = document.getElementById('sync-banner');
        if (banner) {
            banner.style.display = 'block';
            document.getElementById('sync-model-id').innerText = modelId;
            document.getElementById('sync-peer').innerText = sourceNodeId ? sourceNodeId.slice(0, 8) : 'cluster';
            document.getElementById('sync-pct').innerText = 'Starting...';
            document.getElementById('sync-speed').innerText = 'unavailable';
            document.getElementById('sync-progress').style.width = '0%';
        }

        const res = await fetch('/api/mesh/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId, source_node_id: sourceNodeId })
        });
        const job = await res.json();
        closeNodeModal();
    } catch (err) {
        alert('Sync trigger failed: ' + err.message);
    }
}

function openNodeModal(nodeId) {
    if (!clusterTelemetry || !clusterTelemetry.cluster_mesh) return;
    nodeModalTrigger = document.activeElement;
    const mesh = clusterTelemetry.cluster_mesh;
    const allNodes = [mesh.self_node].concat(mesh.peers || []);
    const node = allNodes.find(n => n.node_id === nodeId) || mesh.self_node;

    document.getElementById('modal-node-id').innerText = node.node_id;
    document.getElementById('modal-node-platform').innerText = node.os.toUpperCase() + ' / ' + node.arch;
    document.getElementById('modal-node-tflops').innerText = node.compute_tflops.toFixed(1) + ' TFLOPs';
    document.getElementById('modal-node-bw').innerText = node.bandwidth_gbs.toFixed(1) + ' GB/s';

    // Render Local Models
    const localList = document.getElementById('modal-local-models');
    if (node.available_models && node.available_models.length > 0) {
        localList.innerHTML = node.available_models.map(m => {
            return '<div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1d2828;">' +
                '<div><b style="color:#c4f000; font-family:monospace;">' + m.model_id + '</b>' +
                '<div style="font-size:12px; color:#77817e;">Snapshot: <code>' + (m.revision ? m.revision.slice(0,8) : 'main') + '</code> · ' + m.file_count + ' files</div></div>' +
                '<span class="badge" style="background:#131c16; color:#c4f000; border:1px solid #567000;">' + formatBytesJS(m.total_bytes) + '</span>' +
            '</div>';
        }).join('');
    } else {
        localList.innerHTML = '<div style="color:#77817e; padding:10px 0;">No cached models found on this node.</div>';
    }

    // Render Remote Models available from other peers
    const remoteList = document.getElementById('modal-remote-models');
    const otherNodes = allNodes.filter(n => n.node_id !== node.node_id);
    const remoteMap = new Map();
    otherNodes.forEach(other => {
        (other.available_models || []).forEach(m => {
            const localHas = (node.available_models || []).some(lm => lm.model_id === m.model_id);
            if (!localHas && !remoteMap.has(m.model_id)) {
                remoteMap.set(m.model_id, { model: m, sourceNode: other });
            }
        });
    });

    if (remoteMap.size > 0) {
        let html = '';
        remoteMap.forEach((val, modelId) => {
            html += '<div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1d2828;">' +
                '<div><b style="color:#ebefee; font-family:monospace;">' + modelId + '</b>' +
                '<div style="font-size:12px; color:#77817e;">Available from: <span class="badge" style="background:#111616; border:1px solid #1d2828;">' + val.sourceNode.os.toUpperCase() + '</span> <code>' + val.sourceNode.node_id.slice(0,8) + '</code> (' + formatBytesJS(val.model.total_bytes) + ')</div></div>' +
                '<button class="button" style="color:#c4f000; border-color:#567000;" onclick="triggerSync(\'' + modelId + '\', \'' + val.sourceNode.node_id + '\')">⚡ Sync to this node</button>' +
            '</div>';
        });
        remoteList.innerHTML = html;
    } else {
        remoteList.innerHTML = '<div style="color:#77817e; padding:10px 0;">All cluster models are already synchronized to this node.</div>';
    }

    const modal = document.getElementById('node-modal');
    if (!modal) return;
    modal.style.display = 'flex';
    const close = document.getElementById('node-modal-close');
    if (close) close.focus();
}

function closeNodeModal() {
    const modal = document.getElementById('node-modal');
    if (modal) modal.style.display = 'none';
    if (nodeModalTrigger && typeof nodeModalTrigger.focus === 'function') {
        nodeModalTrigger.focus();
    }
    nodeModalTrigger = null;
}

function nodeModalFocusables(modal) {
    return Array.from(modal.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')).filter(function(element) {
        return !element.disabled && element.getClientRects().length > 0;
    });
}

document.addEventListener('keydown', function(event) {
    const modal = document.getElementById('node-modal');
    if (!modal || modal.style.display !== 'flex') return;
    if (event.key === 'Escape') {
        event.preventDefault();
        closeNodeModal();
        return;
    }
    if (event.key !== 'Tab') return;
    const focusables = nodeModalFocusables(modal);
    if (focusables.length === 0) {
        event.preventDefault();
        modal.focus();
        return;
    }
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
    }
});

function setPromptText(text) {
    const input = document.getElementById('prompt-input');
    if (input) {
        input.value = text;
        input.focus();
    }
}

function executionTargetSelect() {
    let select = document.getElementById('execution-node-select');
    if (select) return select;

    const footer = document.querySelector('#view-playground .composer-footer');
    if (!footer) return null;

    const label = document.createElement('label');
    label.htmlFor = 'execution-node-select';
    label.style.cssText = 'display:flex; align-items:center; gap:6px; font-size:11px; color:#77817e;';
    label.append(document.createTextNode('Run on'));

    select = document.createElement('select');
    select.id = 'execution-node-select';
    select.setAttribute('aria-label', 'Execution node');
    select.style.cssText = 'max-width:260px; height:28px; padding:0 6px; background:#101616; color:#ebefee; border:1px solid #1d2828; border-radius:4px;';
    const local = document.createElement('option');
    local.value = 'local';
    local.textContent = 'This node';
    select.append(local);
    label.append(select);
    footer.insertBefore(label, footer.firstChild);
    return select;
}

function executionTargetText(node) {
    const location = node.local ? 'This node' : 'Peer';
    const platform = node.platform || 'platform unavailable';
    const id = node.node_id ? node.node_id.slice(0, 8) : 'local';
    return location + ' · ' + platform + ' · ' + id;
}

function showExecutionTarget(target, badge) {
    if (!target || !badge) return;
    badge.replaceChildren();
    const prefix = document.createElement('span');
    prefix.style.cssText = 'color:#c4f000; font-weight:600;';
    prefix.textContent = target.local ? 'Execution: this node' : 'Execution: selected peer';
    const detail = document.createElement('span');
    detail.style.marginLeft = '6px';
    detail.textContent = (target.platform || 'platform unavailable') + ' (' + (target.node_id || 'local').slice(0, 8) + ')';
    badge.append(prefix, detail);
    badge.style.display = 'flex';
}

async function refreshExecutionTargets() {
    const select = executionTargetSelect();
    if (!select) return;
    const wanted = select.value || 'local';
    try {
        const res = await fetch('/api/runtime/execution-nodes');
        if (!res.ok) throw new Error('status ' + res.status);
        const data = await res.json();
        select.replaceChildren();

        const self = data.self || { local: true, node_id: 'local', platform: 'local' };
        const local = document.createElement('option');
        local.value = 'local';
        local.textContent = executionTargetText(self);
        select.append(local);

        (data.peers || []).forEach(function(peer) {
            const option = document.createElement('option');
            option.value = peer.node_id;
            option.textContent = executionTargetText(peer) + (peer.available ? '' : ' · unavailable');
            option.disabled = !peer.available;
            select.append(option);
        });
        select.value = Array.from(select.options).some(function(option) { return option.value === wanted && !option.disabled; }) ? wanted : 'local';
        select.title = 'Choose the node that executes this request. Remote nodes return a completed response.';
    } catch (err) {
        select.title = 'Execution targets unavailable: ' + err.message;
    }
}

async function sendPrompt() {
    const input = document.getElementById('prompt-input');
    const out = document.getElementById('output-box');
    const badge = document.getElementById('routing-badge');
    const btn = document.getElementById('gen-btn');
    const targetSelect = executionTargetSelect();
    const prompt = input.value.trim();
    if (!prompt) return;
    const executionNodeID = targetSelect ? targetSelect.value : 'local';
    const remote = executionNodeID !== 'local';

    btn.disabled = true;
    btn.innerText = 'Generating...';
    out.innerText = '';
    if (badge) badge.style.display = 'none';

    try {
        const res = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                max_tokens: 64,
                stream: !remote,
                execution_node_id: executionNodeID
            })
        });
        if (!res.ok) throw new Error(await res.text());
        if (remote) {
            const completion = await res.json();
            out.innerText = completion.choices && completion.choices[0] ? completion.choices[0].message.content : '';
            showExecutionTarget(completion.execution_target, badge);
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed || !trimmed.startsWith('data: ')) continue;
                const payload = trimmed.slice(6);
                if (payload === '[DONE]') break;
                try {
                    const chunk = JSON.parse(payload);
                    const content = chunk.choices[0]?.delta?.content || '';
                    out.innerText += content;
                    showExecutionTarget(chunk.execution_target, badge);
                } catch (err) {}
            }
        }
    } catch (e) {
        out.innerText = 'Error: ' + e.message;
    } finally {
        btn.disabled = false;
        btn.innerText = '↑ Send (⌘ Enter)';
    }
}

async function openInPerfetto() {
    const btn = document.getElementById('btn-perfetto');
    const origText = btn ? btn.innerHTML : '';
    if (btn) btn.innerHTML = '⏳ Loading Perfetto...';

    const win = window.open('https://ui.perfetto.dev', '_blank');
    if (!win) {
        alert('Popup blocked. Please allow popups to launch Perfetto.');
        if (btn) btn.innerHTML = origText;
        return;
    }

    try {
        const resp = await fetch('/debug/perfetto.pftrace');
        if (!resp.ok) {
            // Fall back to gpu profile endpoint if .pftrace not implemented
            const altResp = await fetch('/debug/pprof/gpu');
            const arrayBuf = await altResp.arrayBuffer();
            postTraceToPerfetto(win, arrayBuf, btn, origText);
            return;
        }
        const arrayBuf = await resp.arrayBuffer();
        postTraceToPerfetto(win, arrayBuf, btn, origText);
    } catch (err) {
        console.error('Perfetto launch failed:', err);
        if (btn) btn.innerHTML = origText;
    }
}

function postTraceToPerfetto(win, arrayBuf, btn, origText) {
    const onMessage = (e) => {
        if (e.origin !== 'https://ui.perfetto.dev') return;
        if (e.data === 'PONG') {
            win.postMessage({
                perfetto: {
                    buffer: arrayBuf,
                    title: 'llama-infer Hardware Profile',
                }
            }, 'https://ui.perfetto.dev');
            window.removeEventListener('message', onMessage);
            clearInterval(pingTimer);
            if (btn) btn.innerHTML = origText;
        }
    };

    window.addEventListener('message', onMessage);
    const pingTimer = setInterval(() => {
        win.postMessage('PING', 'https://ui.perfetto.dev');
    }, 100);

    setTimeout(() => {
        clearInterval(pingTimer);
        window.removeEventListener('message', onMessage);
        if (btn) btn.innerHTML = origText;
    }, 10000);
}
