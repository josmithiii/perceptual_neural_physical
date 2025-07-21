"""
Generate Interactive HTML Comparison Interface

Creates an interactive HTML page for comparing target vs reconstruction audio
with parameter displays and navigation between samples.
"""

import json
import os
from pathlib import Path


def generate_html_interface(analysis_file, output_file="comparison.html"):
    """Generate interactive HTML interface from analysis data."""

    with open(analysis_file, "r") as f:
        analysis = json.load(f)

    samples = analysis["samples"]
    param_errors = analysis["parameter_errors"]

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PNP Audio Comparison - Target vs Reconstruction</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .navigation {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
        }}

        .nav-button {{
            padding: 10px 15px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}

        .nav-button:hover {{
            background: #0056b3;
        }}

        .nav-button:disabled {{
            background: #6c757d;
            cursor: not-allowed;
        }}

        .sample-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .audio-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}

        .audio-section {{
            text-align: center;
        }}

        .audio-section h3 {{
            margin-top: 0;
            color: #333;
        }}

        .target {{ border-left: 4px solid #28a745; }}
        .reconstruction {{ border-left: 4px solid #ffc107; }}

        audio {{
            width: 100%;
            margin: 10px 0;
        }}

        .parameters {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .param-section {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }}

        .param-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .param-table th,
        .param-table td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        .param-table th {{
            background-color: #e9ecef;
            font-weight: bold;
        }}

        .error-high {{ color: #dc3545; font-weight: bold; }}
        .error-medium {{ color: #fd7e14; }}
        .error-low {{ color: #28a745; }}

        .summary {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}

        .hidden {{ display: none; }}

        .sample-selector {{
            text-align: center;
            margin: 20px 0;
        }}

        select {{
            padding: 8px 15px;
            font-size: 16px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🥁 PNP Audio Comparison</h1>
        <p>Target vs Reconstruction - Epoch {analysis['epoch']} - {len(samples)} samples</p>

        <div class="summary">
            <h3>Overall Parameter Errors (RMSE)</h3>
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
    """

    for param, error in param_errors.items():
        html_content += f'<div><strong>{param}:</strong> {error["rmse"]:.4f}</div>'

    html_content += f"""
            </div>
        </div>
    </div>

    <div class="sample-selector">
        <label for="sample-select">Select Sample: </label>
        <select id="sample-select" onchange="showSample(this.value)">
    """

    for i, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        html_content += f'<option value="{i}">Sample {sample_id}</option>'

    html_content += """
        </select>
    </div>

    <div class="navigation">
        <button class="nav-button" onclick="prevSample()">← Previous</button>
        <button class="nav-button" onclick="nextSample()">Next →</button>
    </div>
    """

    # Generate sample containers
    for i, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        target_file = sample["target_file"]
        recon_file = sample["reconstruction_file"]
        target_params = sample["target_params"]
        pred_params = sample["predicted_params"]
        errors = sample["param_errors"]

        visibility = "" if i == 0 else "hidden"

        html_content += f"""
    <div id="sample-{i}" class="sample-container {visibility}">
        <h2>Sample {sample_id}</h2>

        <div class="audio-comparison">
            <div class="audio-section target">
                <h3>🎯 Target (Ground Truth)</h3>
                <audio controls preload="metadata">
                    <source src="{target_file}" type="audio/wav">
                    Your browser does not support audio playback.
                </audio>
            </div>

            <div class="audio-section reconstruction">
                <h3>🔄 Reconstruction (Model Prediction)</h3>
                <audio controls preload="metadata">
                    <source src="{recon_file}" type="audio/wav">
                    Your browser does not support audio playback.
                </audio>
            </div>
        </div>

        <div class="parameters">
            <div class="param-section">
                <h4>Target Parameters</h4>
                <table class="param-table">
                    <tr><th>Parameter</th><th>Value</th></tr>
        """

        for param, value in target_params.items():
            html_content += f"<tr><td>{param}</td><td>{value:.4f}</td></tr>"

        html_content += """
                </table>
            </div>

            <div class="param-section">
                <h4>Predicted Parameters</h4>
                <table class="param-table">
                    <tr><th>Parameter</th><th>Value</th><th>Error</th></tr>
        """

        for param, value in pred_params.items():
            error = errors[param]
            error_class = "error-low" if error < 0.1 else ("error-medium" if error < 0.2 else "error-high")
            html_content += f"""<tr><td>{param}</td><td>{value:.4f}</td><td class="{error_class}">{error:.4f}</td></tr>"""

        html_content += """
                </table>
            </div>
        </div>
    </div>
        """

    # Add JavaScript
    html_content += f"""
    <script>
        let currentSample = 0;
        const totalSamples = {len(samples)};

        function showSample(index) {{
            // Hide all samples
            for (let i = 0; i < totalSamples; i++) {{
                document.getElementById('sample-' + i).classList.add('hidden');
            }}

            // Show selected sample
            document.getElementById('sample-' + index).classList.remove('hidden');
            currentSample = parseInt(index);

            // Update selector
            document.getElementById('sample-select').value = index;

            // Pause all audio
            document.querySelectorAll('audio').forEach(audio => audio.pause());
        }}

        function nextSample() {{
            if (currentSample < totalSamples - 1) {{
                showSample(currentSample + 1);
            }}
        }}

        function prevSample() {{
            if (currentSample > 0) {{
                showSample(currentSample - 1);
            }}
        }}

        // Keyboard navigation
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowLeft') {{
                prevSample();
            }} else if (e.key === 'ArrowRight') {{
                nextSample();
            }}
        }});
    </script>
</body>
</html>
    """

    # Write HTML file
    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"Interactive HTML interface generated: {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML comparison interface")
    parser.add_argument("analysis_file", help="Path to analysis.json file")
    parser.add_argument("--output", default="comparison.html",
                       help="Output HTML file (default: comparison.html)")

    args = parser.parse_args()

    generate_html_interface(args.analysis_file, args.output)
