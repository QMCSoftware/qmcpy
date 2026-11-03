# Papers for the kernel/discrepancy collaboration

These unchanged copies of Fred Hickernell's joint papers with Jagadeeswaran
Rathinavel were included at Fred's request for the work in
[issue #646](https://github.com/QMCSoftware/qmcpy/issues/646).
The original publication and copyright notices remain in each PDF. The
QMCPy software license does not relicense these papers.

| Copy | Publication/version | Read first |
| --- | --- | --- |
| [Lattice paper](jagadeeswaran-hickernell-2019-lattice.pdf) | R. Jagadeeswaran and F. J. Hickernell, *Statistics and Computing* 29, 1215–1229 (2019), [DOI](https://doi.org/10.1007/s11222-019-09895-9). Publisher PDF, 15 pages. [Author preprint](https://arxiv.org/abs/1809.09803v2). | Sec. 4.4, "Overcoming cancellation error," printed p. 1224 / PDF page 10; recurrence for the remainder kernel and stable credible-interval formulas (40a–c). |
| [Sobol’ paper](jagadeeswaran-hickernell-2022-sobol.pdf) | R. Jagadeeswaran and F. J. Hickernell, chapter in *Advances in Modeling and Simulation*, 301–318 (2022), [DOI](https://doi.org/10.1007/978-3-031-10193-9_15). Publisher extract with 13 frontmatter pages followed by the 18-page joint chapter. | Sec. 3.3, printed pp. 307–310 / PDF pages 20–23; digital shift invariance and Walsh diagonalization. |

The local source files were in Fred's `Articles_Talks/FinalVersions` library;
no paper text or metadata was edited. Verify future replacements using these
SHA-256 checksums:

```text
7d88df4ae8f403e89002243daf174c5d182e26dca7eb4d8d91dcd2008d32f74d  jagadeeswaran-hickernell-2019-lattice.pdf
bb134da65e6cd0e8ed2ee358f3ce3150604ae0b3a2bead805b2c72f7ff0fb3ce  jagadeeswaran-hickernell-2022-sobol.pdf
```

Jagadeeswaran Rathinavel's separate
[matching-kernels thesis](https://arxiv.org/abs/2210.03253), sec. 6.1, is
additional reading on cancellation for SI and DSI kernels. It is a
single-author thesis, not the joint Sobol’ chapter above.

Return to the [design plan](../kernel-discrepancy-design.md) or
[source audit](../kernel-discrepancy-audit.md).
