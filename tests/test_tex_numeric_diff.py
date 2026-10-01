from memarena.tools.tex_numeric_diff import diff_numbers, extract_numbers

TABLE = r"""
\caption{Mean over $n=3$ seeds.}\label{tab:x}
\begin{tabular}{lc}
% 99.9 commented out
Vanilla & 33.2{\scriptsize$\pm$0.4}\,{\tiny\textcolor{green!55!black}{$\uparrow$2.9}} \\
Oracle & 1{,}579 \\
\end{tabular}
"""


def test_extract_ignores_comments_colours_and_labels() -> None:
    assert extract_numbers(TABLE) == ["3", "33.2", "0.4", "2.9", "1579"]
    assert extract_numbers(TABLE, body_only=True) == ["33.2", "0.4", "2.9", "1579"]


def test_diff_reports_changed_and_missing_numbers() -> None:
    assert diff_numbers(["1.0", "2.0"], ["1.0", "2.0"]) == []
    assert diff_numbers(["1.0", "2.1"], ["1.0", "2.0"]) == [(1, "2.1", "2.0")]
    assert diff_numbers(["1.0"], ["1.0", "2.0"]) == [(1, "<missing>", "2.0")]
    assert diff_numbers(["2.05"], ["2.0"], tol=0.1) == []


def test_skip_header_ignores_rows_before_first_midrule() -> None:
    tex = r"""\begin{tabular}{ll}
\toprule
& \multicolumn{2}{c}{D1} \\ \cmidrule(lr){3-4}
\midrule
A & 1.5 \\
\midrule
B & 2.5 \\
\end{tabular}"""
    assert extract_numbers(tex, skip_header=True) == ["1.5", "2.5"]
