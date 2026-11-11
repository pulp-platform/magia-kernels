# magia-kernels

Kernels for the [MAGIA](https://github.com/pulp-platform/MAGIA) mesh, used by
[magia-sdk](https://github.com/pulp-platform/magia-sdk) as its `kernels` submodule.

- `<kernel>/fp16/spatz/`: fp16 Spatz kernels. Each has a full API, `MAGIA_<kernel>_fp16_spatz`
  (L2 operands, work split over the tiles), and an L1 API, `MAGIA_<kernel>_fp16_spatz_l1`
  (L1 operands, this tile only).
- `mgc_kernels/<kernel>/`: kernels written in [mgc](https://github.com/FrancescoConti/mgc), with
  the generated C next to the `.mgc` source.
- `common/`: helpers shared by the kernels.

They are built and tested from magia-sdk.

## License

Apache License 2.0, see [LICENSE.md](LICENSE.md).
