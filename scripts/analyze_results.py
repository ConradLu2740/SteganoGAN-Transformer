"""实验结果分析脚本：生成对比表格和图表"""
import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')


def load_results(results_path="results/all_results.json"):
    """加载实验结果"""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_ablation_table(results):
    """生成消融实验结果表格（Markdown格式）"""
    lines = []
    lines.append("## 消融实验结果\n")

    lines.append("### A1: 骨干网络消融\n")
    lines.append("| Model | PSNR↑ | SSIM↑ | Bit Acc↑ | Params |")
    lines.append("|-------|-------|-------|----------|--------|")
    ours = results.get("ours_swin_rs_robust_gan", results.get("ours", {}))
    cnn = results.get("A1_cnn", {})
    if ours:
        lines.append(f"| SwinSteg (Ours) | {ours.get('psnr', '[TBD]')} | {ours.get('ssim', '[TBD]')} | "
                     f"{ours.get('bit_accuracy', '[TBD]')} | {ours.get('total_params', '17.1M')} |")
    if cnn:
        lines.append(f"| CNN Baseline | {cnn.get('psnr', '[TBD]')} | {cnn.get('ssim', '[TBD]')} | "
                     f"{cnn.get('bit_accuracy', '[TBD]')} | {cnn.get('total_params', '[TBD]')} |")
    lines.append("")

    lines.append("### A2: Reed-Solomon 纠错码消融\n")
    lines.append("| RS Config | Bit Acc (Clean)↑ | Bit Acc (Robust)↑ |")
    lines.append("|-----------|-----------------|-------------------|")
    no_rs = results.get("A2_no_rs", {})
    if no_rs:
        lines.append(f"| No RS | {no_rs.get('bit_accuracy', '[TBD]')} | [TBD] |")
    if ours:
        lines.append(f"| RS(32) (Ours) | {ours.get('bit_accuracy', '[TBD]')} | [TBD] |")
    lines.append("")

    lines.append("### A3: 鲁棒性训练消融\n")
    lines.append("| Training | Clean Bit Acc↑ | Robust Bit Acc↑ |")
    lines.append("|----------|---------------|-----------------|")
    no_robust = results.get("A3_no_robustness", {})
    if no_robust:
        lines.append(f"| No Robust | {no_robust.get('bit_accuracy', '[TBD]')} | [TBD] |")
    if ours:
        lines.append(f"| Robust (Ours) | {ours.get('bit_accuracy', '[TBD]')} | [TBD] |")
    lines.append("")

    lines.append("### A5: GAN 对抗训练消融\n")
    lines.append("| Training | PSNR↑ | SSIM↑ | Bit Acc↑ |")
    lines.append("|----------|-------|-------|----------|")
    no_gan = results.get("A5_no_gan", {})
    if no_gan:
        lines.append(f"| No GAN | {no_gan.get('psnr', '[TBD]')} | {no_gan.get('ssim', '[TBD]')} | "
                     f"{no_gan.get('bit_accuracy', '[TBD]')} |")
    if ours:
        lines.append(f"| GAN (Ours) | {ours.get('psnr', '[TBD]')} | {ours.get('ssim', '[TBD]')} | "
                     f"{ours.get('bit_accuracy', '[TBD]')} |")
    lines.append("")

    lines.append("### A6: 秘密信息容量消融\n")
    lines.append("| data_depth | Capacity | PSNR↑ | SSIM↑ | Bit Acc↑ |")
    lines.append("|------------|----------|-------|-------|----------|")
    depth_map = {"A6_depth1": (1, "1782B"), "A6_depth2": (2, "3565B"),
                 "ours": (3, "5348B"), "A6_depth4": (4, "7130B")}
    for key, (depth, cap) in depth_map.items():
        r = results.get(key, {})
        if not r and key == "ours":
            r = results.get("ours_swin_rs_robust_gan", {})
        lines.append(f"| {depth} | {cap} | {r.get('psnr', '[TBD]')} | "
                     f"{r.get('ssim', '[TBD]')} | {r.get('bit_accuracy', '[TBD]')} |")
    lines.append("")

    return "\n".join(lines)


def generate_comparison_table(results):
    """生成对比实验结果表格（Markdown格式）"""
    lines = []
    lines.append("## 对比实验结果\n")

    lines.append("### C2: 不同图像尺寸对比\n")
    lines.append("| Size | Capacity | PSNR↑ | SSIM↑ | Bit Acc↑ | Params |")
    lines.append("|------|----------|-------|-------|----------|--------|")
    for key, size, cap in [("C2_64x64", "64×64", "1332B"),
                           ("ours", "128×128", "5348B"),
                           ("C2_256x256", "256×256", "21430B")]:
        r = results.get(key, {})
        if not r and key == "ours":
            r = results.get("ours_swin_rs_robust_gan", {})
        lines.append(f"| {size} | {cap} | {r.get('psnr', '[TBD]')} | "
                     f"{r.get('ssim', '[TBD]')} | {r.get('bit_accuracy', '[TBD]')} | "
                     f"{r.get('total_params', '[TBD]')} |")
    lines.append("")

    lines.append("### C3: 不同 RS 纠错强度对比\n")
    lines.append("| RS Config | nsym | Bit Acc (Clean)↑ | Capacity Loss |")
    lines.append("|-----------|------|-----------------|---------------|")
    for key, nsym, cap_loss in [("A2_no_rs", "0", "0%"),
                                 ("C3_rs8", "8", "3.1%"),
                                 ("C3_rs16", "16", "6.3%"),
                                 ("ours", "32", "12.5%"),
                                 ("C3_rs64", "64", "25.1%")]:
        r = results.get(key, {})
        if not r and key == "ours":
            r = results.get("ours_swin_rs_robust_gan", {})
        lines.append(f"| RS({nsym}) | {nsym} | {r.get('bit_accuracy', '[TBD]')} | {cap_loss} |")
    lines.append("")

    return "\n".join(lines)


def generate_summary(results):
    """生成实验总结"""
    lines = []
    lines.append("## 实验总结\n")

    ours = results.get("ours_swin_rs_robust_gan", results.get("ours", {}))
    if ours:
        lines.append(f"**基准模型 (SwinSteg)**:")
        lines.append(f"- PSNR: {ours.get('psnr', '[TBD]')} dB")
        lines.append(f"- SSIM: {ours.get('ssim', '[TBD]')}")
        lines.append(f"- Bit Accuracy: {ours.get('bit_accuracy', '[TBD]')}")
        lines.append(f"- 总参数量: {ours.get('total_params', '[TBD]'):,}")
        lines.append(f"- 训练时间: {ours.get('training_time_seconds', '[TBD]')}s")
        lines.append("")

    comparisons = []
    for key, label in [("A1_cnn", "CNN vs Swin"),
                       ("A2_no_rs", "RS纠错贡献"),
                       ("A3_no_robustness", "鲁棒性训练贡献"),
                       ("A5_no_gan", "GAN对抗训练贡献")]:
        r = results.get(key, {})
        if r and ours:
            diff = ours.get("bit_accuracy", 0) - r.get("bit_accuracy", 0)
            comparisons.append((label, diff))

    if comparisons:
        lines.append("**各创新点贡献排序 (Bit Accuracy 提升)**:")
        comparisons.sort(key=lambda x: x[1], reverse=True)
        for label, diff in comparisons:
            lines.append(f"- {label}: {diff*100:+.1f}%")
        lines.append("")

    return "\n".join(lines)


def generate_latex_table(results):
    """生成LaTeX格式的消融实验表格"""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation Study Results}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("Model & PSNR (dB) & SSIM & Bit Acc (\\%) & Params \\\\")
    lines.append("\\hline")

    ours = results.get("ours_swin_rs_robust_gan", results.get("ours", {}))
    if ours:
        lines.append(f"SwinSteg (Ours) & {ours.get('psnr', 'TBD')} & {ours.get('ssim', 'TBD')} & "
                     f"{ours.get('bit_accuracy', 'TBD')} & {ours.get('total_params', '17.1M')} \\\\")

    for key, label in [("A1_cnn", "CNN Baseline"), ("A2_no_rs", "w/o RS"),
                       ("A3_no_robustness", "w/o Robust"), ("A5_no_gan", "w/o GAN")]:
        r = results.get(key, {})
        if r:
            lines.append(f"{label} & {r.get('psnr', 'TBD')} & {r.get('ssim', 'TBD')} & "
                         f"{r.get('bit_accuracy', 'TBD')} & {r.get('total_params', 'TBD')} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="实验结果分析脚本")
    parser.add_argument("--results", type=str, default="results/all_results.json",
                        help="结果JSON文件路径")
    parser.add_argument("--format", type=str, default="all",
                        choices=["markdown", "latex", "all"],
                        help="输出格式")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（默认打印到控制台）")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"结果文件不存在: {args.results}")
        print("请先运行: python scripts/run_experiments.py --experiment all")
        return

    results = load_results(args.results)

    output_parts = []

    if args.format in ("markdown", "all"):
        md_content = generate_ablation_table(results) + "\n" + \
                     generate_comparison_table(results) + "\n" + \
                     generate_summary(results)
        output_parts.append(md_content)

    if args.format in ("latex", "all"):
        latex_content = generate_latex_table(results)
        output_parts.append("\n## LaTeX Tables\n\n```latex\n" + latex_content + "\n```")

    full_output = "\n".join(output_parts)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(full_output)
        print(f"结果已保存到: {args.output}")
    else:
        print(full_output)


if __name__ == "__main__":
    main()
