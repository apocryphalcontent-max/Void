# Void-State Tools Deployment Guide

## Table of Contents
1. [Local Development Setup](#local-development-setup)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring & Observability](#monitoring--observability)
6. [Production Checklist](#production-checklist)

---

## Local Development Setup

### Prerequisites
- Python 3.9+
- pip or poetry
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/apocryphalcontent-max/Messy.git
cd Messy

# Install dependencies
pip install -e .

# Or with poetry
poetry install

# Verify installation
python -c "import void_state_tools; print(void_state_tools.__version__)"
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest void_state_tools/tests/ -v

# Run with coverage
pytest void_state_tools/tests/ --cov=void_state_tools --cov-report=html

# Run specific test file
pytest void_state_tools/tests/test_mvp.py -v
```

### Running Benchmarks

```bash
# Run performance benchmarks
python -m void_state_tools.benchmarks

# Expected output: < 100ns per hook execution
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY void_state_tools/ /app/void_state_tools/
COPY VOID_STATE_*.md /app/
COPY setup.py /app/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Install monitoring tools
RUN pip install prometheus-client

# Expose metrics port
EXPOSE 9090

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV VOID_STATE_LOG_LEVEL=INFO

# Run application
CMD ["python", "-m", "void_state_tools"]
```

### Build and Run

```bash
# Build image
docker build -t void-state-tools:latest .

# Run container
docker run -d \
  --name void-state-tools \
  -p 9090:9090 \
  -v $(pwd)/config:/app/config \
  -e VOID_STATE_CONFIG=/app/config/registry_config.json \
  void-state-tools:latest

# View logs
docker logs -f void-state-tools

# Stop container
docker stop void-state-tools
```

### Docker Compose

```yaml
version: '3.8'

services:
  void-state-tools:
    build: .
    container_name: void-state-tools
    ports:
      - "9090:9090"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    environment:
      - VOID_STATE_CONFIG=/app/config/registry_config.json
      - VOID_STATE_LOG_LEVEL=INFO
      - PROMETHEUS_PORT=9090
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import void_state_tools; print('ok')"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    depends_on:
      - void-state-tools
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus

volumes:
  prometheus-data:
  grafana-data:
```

Run with:
```bash
docker-compose up -d
```

---

## Kubernetes Deployment

### Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: void-state-tools
```

### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: void-state-config
  namespace: void-state-tools
data:
  registry_config.json: |
    {
      "registry_config": {
        "max_tools": 100,
        "allow_dynamic_registration": true
      },
      "monitoring": {
        "enabled": true,
        "prometheus_port": 9090
      }
    }
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: void-state-tools
  namespace: void-state-tools
spec:
  replicas: 3
  selector:
    matchLabels:
      app: void-state-tools
  template:
    metadata:
      labels:
        app: void-state-tools
    spec:
      containers:
      - name: void-state-tools
        image: void-state-tools:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 9090
          name: metrics
        env:
        - name: VOID_STATE_CONFIG
          value: /config/registry_config.json
        - name: VOID_STATE_LOG_LEVEL
          value: INFO
        volumeMounts:
        - name: config
          mountPath: /config
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 9090
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: void-state-config
```

### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: void-state-tools
  namespace: void-state-tools
spec:
  selector:
    app: void-state-tools
  ports:
  - port: 9090
    targetPort: 9090
    name: metrics
  type: ClusterIP
```

### ServiceMonitor (Prometheus Operator)

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: void-state-tools
  namespace: void-state-tools
spec:
  selector:
    matchLabels:
      app: void-state-tools
  endpoints:
  - port: metrics
    interval: 30s
```

### Deploy

```bash
# Apply all resources
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f servicemonitor.yaml

# Check status
kubectl get pods -n void-state-tools
kubectl logs -f deployment/void-state-tools -n void-state-tools

# Scale deployment
kubectl scale deployment void-state-tools --replicas=5 -n void-state-tools
```

---

## Cloud Deployment

### AWS (ECS)

```bash
# Create ECR repository
aws ecr create-repository --repository-name void-state-tools

# Build and push image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t void-state-tools .
docker tag void-state-tools:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/void-state-tools:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/void-state-tools:latest

# Create ECS task definition and service (use AWS Console or CLI)
```

### GCP (Cloud Run)

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/<project-id>/void-state-tools

# Deploy to Cloud Run
gcloud run deploy void-state-tools \
  --image gcr.io/<project-id>/void-state-tools \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 2 \
  --port 9090
```

### Azure (Container Instances)

```bash
# Create resource group
az group create --name void-state-tools --location eastus

# Create container instance
az container create \
  --resource-group void-state-tools \
  --name void-state-tools \
  --image void-state-tools:latest \
  --cpu 2 \
  --memory 4 \
  --ports 9090 \
  --environment-variables VOID_STATE_LOG_LEVEL=INFO
```

---

## Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'void-state-tools'
    static_configs:
      - targets: ['void-state-tools:9090']
    metrics_path: '/metrics'
```

### Grafana Dashboard

Import the pre-built dashboard:
```json
{
  "dashboard": {
    "title": "Void-State Tools",
    "panels": [
      {
        "title": "Active Tools",
        "targets": [{"expr": "void_state_active_tools"}]
      },
      {
        "title": "Hook Execution Time (P95)",
        "targets": [{"expr": "histogram_quantile(0.95, rate(void_state_hook_duration_seconds_bucket[5m]))"}]
      },
      {
        "title": "Tool Memory Usage",
        "targets": [{"expr": "void_state_tool_memory_bytes"}]
      }
    ]
  }
}
```

### Logs

```bash
# View logs
kubectl logs -f deployment/void-state-tools -n void-state-tools

# Docker logs
docker logs -f void-state-tools

# Filter errors
kubectl logs deployment/void-state-tools -n void-state-tools | grep ERROR
```

### Alerting

```yaml
# alerting-rules.yml
groups:
  - name: void-state-tools
    rules:
      - alert: ToolHighMemoryUsage
        expr: void_state_tool_memory_bytes > 500000000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Tool {{ $labels.tool_name }} using excessive memory"
      
      - alert: HookOverheadExceeded
        expr: void_state_hook_duration_seconds > 0.001
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Hook {{ $labels.hook_name }} exceeding budget"
```

---

## Production Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Performance benchmarks meet requirements
- [ ] Security scan completed
- [ ] Resource limits configured
- [ ] Monitoring configured
- [ ] Alerts configured
- [ ] Backup strategy defined

### Deployment
- [ ] Deploy to staging first
- [ ] Run smoke tests
- [ ] Monitor metrics for 24 hours
- [ ] Gradual rollout (canary/blue-green)
- [ ] Keep rollback plan ready

### Post-Deployment
- [ ] Monitor error rates
- [ ] Check resource usage
- [ ] Verify hook performance
- [ ] Review logs for anomalies
- [ ] Update documentation

### Ongoing Maintenance
- [ ] Weekly metric reviews
- [ ] Monthly performance audits
- [ ] Quarterly capacity planning
- [ ] Regular security updates
- [ ] Keep dependencies updated

---

## Troubleshooting

### Common Issues

**High Memory Usage**
```bash
# Check tool memory
kubectl top pods -n void-state-tools

# Restart pod
kubectl delete pod <pod-name> -n void-state-tools
```

**Hook Performance Issues**
```bash
# Run benchmarks
docker exec void-state-tools python -m void_state_tools.benchmarks

# Check overhead
kubectl logs deployment/void-state-tools -n void-state-tools | grep "overhead"
```

**Connection Issues**
```bash
# Test service
kubectl port-forward svc/void-state-tools 9090:9090 -n void-state-tools
curl http://localhost:9090/metrics
```

For more help, see the API documentation at `void_state_tools/docs/API.md`.
