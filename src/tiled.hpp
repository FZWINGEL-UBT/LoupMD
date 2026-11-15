#pragma once
// The tiled capacity mode (PLAN.md Stage 5). A cell too large for one whole-system call is evaluated tile by tile:
// a tile's local atoms receive, and the atoms within cutoff plus skin of them send as ghosts. This header holds
// the plan of the tiles, made on the host (tile_plan.cu, plan task S5-T01).
#include <cstdint>
#include <vector>

namespace loup {

struct TilePlan {
    int grid[3] = {1, 1, 1};            // blocks per cell axis
    std::vector<int64_t> local_ptr;     // [tiles + 1] where each tile's locals start in `local`
    std::vector<int32_t> local;         // [atoms] every atom once, tile by tile; ascending within a tile
    std::vector<int32_t> local_shift;   // [3 per local] cell vectors to add to the atom's position: it is then
                                        // inside the cell, which is the frame its tile works in
    std::vector<int64_t> ghost_ptr;     // [tiles + 1] where each tile's ghost candidates start in `ghost`
    std::vector<int32_t> ghost;         // the atom of each candidate
    std::vector<int32_t> ghost_shift;   // [3 per candidate] cell vectors to add to that atom's position: the image
                                        // that lies within `reach` of the tile, in the same frame as its locals
    int64_t tiles() const { return static_cast<int64_t>(local_ptr.size()) - 1; }
};

// The tiles of `n` atoms at `positions` [n, 3] in `cell` (rows are the lattice vectors, Angstrom).
//
// Atoms are binned by their wrapped fractional coordinates on a grid of blocks. `shape` gives the blocks per axis;
// with shape == nullptr the planner chooses: from the cell's plane spacings, so that a block is about cubic and
// holds about `receivers` atoms, and of that grid and six finer ones the grid whose blocks make the fewest tiles.
// A block above a cap is split into consecutive pieces in atom order: at most `receivers` atoms, and with
// `degree` (each atom's edge count) at most `edges` edges, a single atom excepted.
//
// reach > 0 (cutoff plus skin): each tile also lists its ghost candidates, every image of an atom whose fractional
// coordinates lie within reach of the tile's bounding box (a plane's distance bounds every distance across it),
// so every atom within `reach` of a local is in the tile, as a local or as a candidate. A candidate may be farther
// than `reach` from every local (the box's corners); the tile's neighbour list drops those.
//
// Throws std::runtime_error with the reason when nothing can be planned.
TilePlan plan_tiles(const double* positions, int64_t n, const double* cell, int receivers, int64_t edges,
                    const int32_t* degree, double reach, const int* shape);

}  // namespace loup
