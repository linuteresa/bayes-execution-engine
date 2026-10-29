"""Export a real span; confirm its receipt in the configured collector's logs."""

from core.telemetry import configure_telemetry, log_event, span


def main():
    provider = configure_telemetry()
    if provider is None:
        raise RuntimeError("set OTEL_EXPORTER_OTLP_ENDPOINT before the collector smoke check")
    try:
        with span("validation.collector", run_id="live-validation"):
            log_event("validation.export")
        if not provider.force_flush(timeout_millis=10000):
            raise TimeoutError("trace export timed out")
    finally:
        provider.shutdown()


if __name__ == "__main__":
    main()
