import json
from datetime import datetime
from pathlib import Path

def track_quality():
    """Track docstring quality metrics over time."""
    
    # Run cleanup in dry-run mode
    import subprocess
    result = subprocess.run(
        ['python', 'scripts/cleanup_docstrings.py', 'src/', '--dry-run', '--report', '--output', 'temp_report.json'],
        capture_output=True
    )
    
    # Load report
    with open('temp_report.json') as f:
        report = json.load(f)
    
    # Calculate metrics
    total_files = report['total_files']
    needs_cleanup = len(report['files_modified'])
    quality_score = (1 - needs_cleanup / total_files) * 100 if total_files > 0 else 100
    
    # Save metrics
    metrics = {
        'date': datetime.now().isoformat(),
        'quality_score': quality_score,
        'total_files': total_files,
        'needs_cleanup': needs_cleanup
    }
    
    # Append to history
    history_file = Path('.metrics/docstring_quality.json')
    history_file.parent.mkdir(exist_ok=True)
    
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
    else:
        history = []
    
    history.append(metrics)
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"📊 Docstring Quality Score: {quality_score:.1f}%")
    return quality_score

if __name__ == "__main__":
    track_quality()