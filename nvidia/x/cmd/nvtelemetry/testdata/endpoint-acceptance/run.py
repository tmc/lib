#!/usr/bin/env python3
"""Controlled mock CLI acceptance against pinned real OTLP/Prometheus backends.

Usage: python3 run.py --output ~/tmp/lib-bpf/<date>-endpoint-acceptance --seconds 60
Requires Go, Docker and openssl. No credentials/hardware are discovered.
Owned containers/network and private keys are removed even on failure.
"""
import argparse
import hashlib
import json
import pathlib
import secrets
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

OTEL = 'ghcr.io/open-telemetry/opentelemetry-collector-releases/opentelemetry-collector@sha256:310a800ad69ee430e7c541796852a242c9c7db97aaad4daa5ccf843c525fbdb2'
PROM = 'quay.io/prometheus/prometheus@sha256:49214755b6153f90a597adcbff0252cc61069f8ab69ce8411285cd4a560e8038'
UBUNTU = 'ubuntu@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55'
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--seconds', type=int, default=60)
a = parser.parse_args()
if a.seconds < 10 or a.seconds > 3600:
    parser.error('seconds must be 10..3600')
out = pathlib.Path(a.output).expanduser().resolve()
out.mkdir(parents=True, exist_ok=False)
private = out / 'private'
private.mkdir(mode=0o700)
repo = pathlib.Path(__file__).resolve().parents[6]
module = repo / 'nvidia'
name = 'lib-telemetry-endpoints-' + secrets.token_hex(6)
containers = []
network_created = False
result = {'status': 'FAIL', 'claim': 'controlled synthetic CLI endpoint/backend acceptance only',
          'seconds': a.seconds, 'images': {'otel': OTEL, 'prometheus': PROM, 'runner': UBUNTU}}

def command(*args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, stderr=subprocess.STDOUT, text=True).strip()

def docker(*args):
    return command('docker', *args)

def start(label, image, args, ports=()):
    n = name + '-' + label
    containers.append(n)
    options = ['run', '-d', '--name', n, '--network', name, '--network-alias', label,
               '--user', '0:0', '-v', str(private) + ':/conf:ro']
    for port in ports:
        options += ['-p', '127.0.0.1::' + str(port)]
    docker(*options, image, *args)
    return n

def get(url):
    with urllib.request.urlopen(url, timeout=3) as r:
        return r.read()

try:
    result['harness_sha256'] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    result['source'] = command('git', 'rev-parse', 'HEAD', cwd=repo)
    arch = {'aarch64': 'arm64', 'x86_64': 'amd64'}[docker('info', '--format', '{{.Architecture}}')]
    result['architecture'] = arch
    command('env', 'CGO_ENABLED=0', 'GOOS=linux', 'GOARCH=' + arch,
            'go', 'build', '-o', str(private / 'nvtelemetry'), './x/cmd/nvtelemetry', cwd=module)
    command('openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
            '-subj', '/CN=telemetry', '-addext', 'subjectAltName=DNS:telemetry',
            '-keyout', str(private / 'key.pem'), '-out', str(private / 'cert.pem'))
    result['certificate_sha256'] = hashlib.sha256((private / 'cert.pem').read_bytes()).hexdigest()
    token = secrets.token_hex(32)
    (private / 'token').write_text(token)
    for signal in ('metrics', 'logs'):
        (private / (signal + '.json')).write_text(json.dumps({
            'url': 'http://collector:4318/v1/' + signal, 'allow_http': True,
            'timeout_nanos': 5000000000, 'max_bytes': 65536}))
    (private / 'collector.yaml').write_text('''receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
exporters:
  debug:
    verbosity: detailed
  prometheus:
    endpoint: 0.0.0.0:8889
    translation_strategy: UnderscoreEscapingWithoutSuffixes
    enable_open_metrics: true
    send_timestamps: false
service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [debug, prometheus]
    logs:
      receivers: [otlp]
      exporters: [debug]
''')
    (private / 'prometheus.yaml').write_text('''global:
  scrape_interval: 1s
  scrape_timeout: 500ms
scrape_configs:
  - job_name: telemetry_direct
    scheme: https
    authorization:
      type: Bearer
      credentials_file: /conf/token
    tls_config:
      ca_file: /conf/cert.pem
      server_name: telemetry
    static_configs:
      - targets: ['telemetry:9464']
  - job_name: telemetry_otlp
    honor_labels: true
    static_configs:
      - targets: ['collector:8889']
''')
    # Controlled fixture containers run as root to read mode-0600 test credentials.
    # This harness is not a production deployment qualification.
    private.chmod(0o700)
    for p in private.iterdir():
        p.chmod(0o755 if p.name == 'nvtelemetry' else 0o600)
    docker('network', 'create', name)
    network_created = True
    collector = start('collector', OTEL, ['--config=/conf/collector.yaml'])
    prom = start('prometheus', PROM, ['--config.file=/conf/prometheus.yaml', '--storage.tsdb.path=/prometheus'], [9090])
    # Collector admission: start the producer only after the receiver is listening.
    deadline = time.monotonic() + 30
    while 'Everything is ready.' not in docker('logs', collector):
        if time.monotonic() > deadline:
            raise RuntimeError('collector startup deadline')
        time.sleep(.25)
    telemetry = start('telemetry', UBUNTU, ['/conf/nvtelemetry', 'serve', '-mock',
        '-listen', '0.0.0.0:9464', '-allow-remote', '-tls-cert', '/conf/cert.pem',
        '-tls-key', '/conf/key.pem', '-auth-token-file', '/conf/token',
        '-interval', '100ms', '-budget', str(a.seconds * 20 + 1000),
        '-otlp-metrics-config', '/conf/metrics.json', '-otlp-logs-config', '/conf/logs.json'])
    base = 'http://' + docker('port', prom, '9090/tcp')
    start_time = time.monotonic()
    samples = []
    deadline = start_time + a.seconds
    while time.monotonic() < deadline:
        try:
            raw = get(base + '/api/v1/query?' + urllib.parse.urlencode({'query': '{__name__=~"nvtelemetry_.*"}'}))
            data = json.loads(raw)
            if data['status'] != 'success':
                raise RuntimeError('Prometheus query failed')
            samples.append(data)
            # Any early producer exit invalidates the soak.
            if docker('inspect', '--format', '{{.State.Running}}', telemetry) != 'true':
                raise RuntimeError('producer exited before soak completed')
        except urllib.error.URLError:
            if time.monotonic() - start_time > 10:
                raise
        time.sleep(1)
    (out / 'queries.json').write_text(json.dumps(samples, indent=2))
    rows = samples[-1]['data']['result']
    for job in ('telemetry_direct', 'telemetry_otlp'):
        selected = [r for r in rows if r['metric'].get('job') == job and
                    r['metric']['__name__'] == 'nvtelemetry_power_milliwatts']
        if len(selected) != 1 or float(selected[0]['value'][1]) != 30000:
            raise RuntimeError('missing or wrong synthetic power series: ' + job)
        if 'mock' not in str(selected[0]['metric']) or 'derived' not in str(selected[0]['metric']):
            raise RuntimeError('mock/derived provenance lost: ' + job)
    targets = get(base + '/api/v1/targets')
    (out / 'targets.json').write_bytes(targets)
    if any(t['health'] != 'up' for t in json.loads(targets)['data']['activeTargets']):
        raise RuntimeError('scrape target down')
    docker('stop', '-t', '10', telemetry)
    if docker('inspect', '--format', '{{.State.ExitCode}}', telemetry) != '0':
        raise RuntimeError('producer drain returned failure')
    producer_lines = docker('logs', telemetry).splitlines()
    summaries = {entry['signal']: entry['stats'] for entry in
                 (json.loads(line) for line in producer_lines if line.startswith('{'))}
    for signal in ('metrics', 'logs'):
        stats = summaries[signal]
        if stats['EnqueuedBatches'] == 0 or stats['SucceededBatches'] != stats['EnqueuedBatches'] or stats['FailedBatches'] or stats['PendingBatches'] or stats['DroppedBatches']:
            raise RuntimeError('incomplete signal statistics: ' + signal)
    if summaries['metrics']['EnqueuedBatches'] != summaries['logs']['EnqueuedBatches']:
        raise RuntimeError('unpaired signal admission')
    result['delivery_stats'] = summaries
    logs = docker('logs', collector)
    if 'nvtelemetry.logs.v1' not in logs or 'GPU-mock' not in logs:
        raise RuntimeError('actual collector did not decode diagnostic logs')
    if token in logs:
        raise RuntimeError('authentication leaked into backend logs')
    result.update(status='PASS', scrape_targets=2, query_snapshots=len(samples),
                  paired_signals=True, producer_exit=0, trusted_tls=True,
                  elapsed_seconds=time.monotonic() - start_time)
except Exception as e:
    result['error'] = type(e).__name__ + ': ' + str(e)
    raise
finally:
    for n in containers:
        try:
            (out / (n + '-logs.txt')).write_text(docker('logs', n))
            docker('stop', '-t', '10', n)
            (out / (n + '-inspect.json')).write_text(docker('inspect', n))
            docker('rm', n)
        except Exception as e:
            (out / (n + '-cleanup-error.txt')).write_text(str(e))
    if network_created:
        try:
            docker('network', 'rm', name)
        except Exception as e:
            (out / 'network-cleanup-error.txt').write_text(str(e))
    shutil.rmtree(private)
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    (out / 'hashes.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir() if p.is_file()}, indent=2))
    print(out)
