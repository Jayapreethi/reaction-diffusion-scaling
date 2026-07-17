.PHONY: help benchmark benchmark-cpu benchmark-cpu-simple benchmark-cluster benchmark-local-only results clean docs

# Fisher-KPP GPU/CPU Benchmark Pipeline
# National Lab Standard for Reproducible Computational Science
#
# Common targets:
#   make benchmark          - Full pipeline (CPU local + GPU cluster)
#   make benchmark-cpu      - CPU benchmarks only (via orchestrator)
#   make benchmark-cpu-simple - CPU benchmarks only (simple NumPy, no PyTorch)
#   make benchmark-cluster  - Submit cluster jobs only
#   make results            - Aggregate results from latest run
#   make docs               - Generate documentation
#   make clean              - Clean output files

help:
	@echo "Fisher-KPP Benchmark Pipeline"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  benchmark              Full pipeline (CPU + cluster GPU)"
	@echo "  benchmark-cpu          CPU benchmarks (full orchestrator)"
	@echo "  benchmark-cpu-simple   CPU benchmarks (pure NumPy, no GPU/PyTorch)"
	@echo "  benchmark-cluster      Submit GPU jobs to Talon (no local CPU)"
	@echo "  results                Aggregate and report latest results"
	@echo "  docs                   Generate pipeline documentation"
	@echo "  clean                  Clean output directories"
	@echo "  help                   Show this message"
	@echo ""
	@echo "Configuration: config/benchmark_config.yaml"
	@echo "Results: outputs/benchmark_*/"

benchmark:
	@echo "=========================================="
	@echo "FULL PIPELINE: CPU BENCHMARKS + CLUSTER GPU JOBS"
	@echo "=========================================="
	python3 scripts/run_benchmark_suite.py --config config/benchmark_config.yaml

benchmark-cpu:
	@echo "=========================================="
	@echo "CPU BENCHMARKS ONLY (Local Machine)"
	@echo "=========================================="
	python3 scripts/run_benchmark_suite.py --local-only --config config/benchmark_config.yaml
	@echo ""
	@echo "Check outputs/ for results"

benchmark-cpu-simple:
	@echo "=========================================="
	@echo "CPU BENCHMARKS (Pure NumPy, No PyTorch)"
	@echo "=========================================="
	python3 scripts/cpu_benchmark.py --config config/benchmark_config.yaml
	@echo ""
	@echo "Check outputs/ for results"

benchmark-cluster:
	@echo "=========================================="
	@echo "SUBMIT GPU JOBS TO TALON CLUSTER"
	@echo "=========================================="
	python3 scripts/run_benchmark_suite.py --cluster-only --config config/benchmark_config.yaml
	@echo ""
	@echo "Monitor jobs with: squeue -u jayapreethi.mohan"

results:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "AGGREGATING LATEST RESULTS"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@LATEST=$$(ls -td outputs/benchmark_*/ | head -1) && \
	if [ -n "$$LATEST" ]; then \
		echo "Processing: $$LATEST"; \
		python3 scripts/aggregate_results.py --run-dir "$$LATEST"; \
	else \
		echo "❌ No benchmark results found in outputs/"; \
	fi

docs:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "PIPELINE DOCUMENTATION"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Configuration:"
	@echo "  📋 config/benchmark_config.yaml"
	@echo ""
	@echo "Scripts:"
	@echo "  🎯 scripts/run_benchmark_suite.py      - Master orchestrator"
	@echo "  📊 scripts/gpu_benchmark_single.py    - GPU benchmark (Talon)"
	@echo "  📈 scripts/aggregate_results.py       - Result aggregation"
	@echo ""
	@echo "Output directories:"
	@echo "  📁 outputs/benchmark_*/              - Run results with timestamps"
	@echo "  📁 outputs/benchmark_*/slurm_jobs/   - Generated SLURM scripts"
	@echo "  📁 outputs/benchmark_*/metadata.json - Environment capture"
	@echo ""
	@echo "For details, see:"
	@echo "  - docs/GPU_GUIDE.md (GPU acceleration guide)"
	@echo "  - docs/ARCHITECTURE.md (System design)"
	@echo "  - TECH_BLOG_POST.md (Why GPU is slow for small problems)"

clean:
	@echo "Cleaning benchmark outputs..."
	rm -rf outputs/benchmark_*/
	@echo "✓ Cleaned"

.PHONY: status
status:
	@echo "Pipeline Status:"
	@echo ""
	@ls -td outputs/benchmark_*/ 2>/dev/null | head -3 | while read dir; do \
		echo "Run: $$(basename $$dir)"; \
		if [ -f "$$dir/metadata.json" ]; then \
			echo "  ✓ Environment captured"; \
		fi; \
		if [ -f "$$dir/cpu_results.json" ]; then \
			echo "  ✓ CPU results available"; \
		fi; \
		if [ -f "$$dir/BENCHMARK_REPORT.md" ]; then \
			echo "  ✓ Report generated"; \
		fi; \
		echo ""; \
	done
