from src.report_generator import generate_report

metrics = {
    "statistical_parity": -0.18,
    "equal_opportunity": -0.12,
    "false_positive_rate": 0.09
}

generate_report(metrics, "reports/audit_report.pdf")
print("Report generated")
