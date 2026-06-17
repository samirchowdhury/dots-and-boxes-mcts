#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Board {
    int rows = 0;
    int cols = 0;
    int action_count = 0;
    int total_boxes = 0;
    std::vector<std::string> action_ids;
    std::vector<int> edge_box_a;
    std::vector<int> edge_box_b;
    std::unordered_map<std::string, int> move_to_index;
};

struct State {
    int current_player = 0;
    int edge_count = 0;
    std::vector<int8_t> edges;
    std::vector<int8_t> boxes;
    int scores[2] = {0, 0};
};

struct Node {
    State state;
    int move = -1;
    double prior = 1.0;
    std::vector<int> children;
    int visits = 0;
    double value_sum = 0.0;
    bool expanded = false;
};

struct EvalResult {
    std::vector<double> priors;
    double value = 0.0;
};

struct PathEdge {
    int parent = -1;
    int child = -1;
};

struct Reservation {
    std::vector<PathEdge> path;
    int leaf = -1;
    int leaf_player = 0;
    bool terminal = false;
    double terminal_value = 0.0;
    int eval_index = -1;
};

struct SearchOptions {
    int simulations = 1;
    double c_puct = 1.5;
    unsigned int seed = 1;
    double root_dirichlet_alpha = 0.0;
    double root_exploration_fraction = 0.0;
    int batch_size = 1;
    double virtual_loss = 1.0;
};

PyObject* new_ref(PyObject* object) {
    Py_XINCREF(object);
    return object;
}

bool set_item_steals(PyObject* dict, const char* key, PyObject* value) {
    if (value == nullptr) {
        return false;
    }
    const int result = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return result == 0;
}

PyObject* borrowed_dict_item(PyObject* dict, const char* key) {
    PyObject* value = PyDict_GetItemString(dict, key);
    if (value == nullptr) {
        PyErr_Format(PyExc_KeyError, "missing snapshot key '%s'", key);
    }
    return value;
}

long long_from_object(PyObject* value, const char* context) {
    long parsed = PyLong_AsLong(value);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError, "expected integer for %s", context);
        return 0;
    }
    return parsed;
}

Board make_board(int rows, int cols) {
    if (rows < 2 || cols < 2) {
        throw std::invalid_argument("Dots and Boxes needs at least a 2x2 dot grid.");
    }

    Board board;
    board.rows = rows;
    board.cols = cols;
    board.total_boxes = (rows - 1) * (cols - 1);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols - 1; ++col) {
            board.action_ids.push_back(
                "h:" + std::to_string(row) + ":" + std::to_string(col)
            );
        }
    }
    for (int row = 0; row < rows - 1; ++row) {
        for (int col = 0; col < cols; ++col) {
            board.action_ids.push_back(
                "v:" + std::to_string(row) + ":" + std::to_string(col)
            );
        }
    }

    board.action_count = static_cast<int>(board.action_ids.size());
    board.edge_box_a.assign(board.action_count, -1);
    board.edge_box_b.assign(board.action_count, -1);
    for (int index = 0; index < board.action_count; ++index) {
        board.move_to_index[board.action_ids[index]] = index;
    }

    const int horizontal_count = rows * (cols - 1);
    for (int index = 0; index < board.action_count; ++index) {
        if (index < horizontal_count) {
            const int row = index / (cols - 1);
            const int col = index - row * (cols - 1);
            int slot = 0;
            if (row > 0) {
                board.edge_box_a[index] = (row - 1) * (cols - 1) + col;
                slot = 1;
            }
            if (row < rows - 1) {
                if (slot == 0) {
                    board.edge_box_a[index] = row * (cols - 1) + col;
                } else {
                    board.edge_box_b[index] = row * (cols - 1) + col;
                }
            }
        } else {
            const int offset = index - horizontal_count;
            const int row = offset / cols;
            const int col = offset - row * cols;
            int slot = 0;
            if (col > 0) {
                board.edge_box_a[index] = row * (cols - 1) + (col - 1);
                slot = 1;
            }
            if (col < cols - 1) {
                if (slot == 0) {
                    board.edge_box_a[index] = row * (cols - 1) + col;
                } else {
                    board.edge_box_b[index] = row * (cols - 1) + col;
                }
            }
        }
    }

    return board;
}

int box_edge_count(const State& state, const Board& board, int box_index) {
    const int box_cols = board.cols - 1;
    const int row = box_index / box_cols;
    const int col = box_index - row * box_cols;
    const int top = row * (board.cols - 1) + col;
    const int bottom = (row + 1) * (board.cols - 1) + col;
    const int vertical_start = board.rows * (board.cols - 1);
    const int left = vertical_start + row * board.cols + col;
    const int right = vertical_start + row * board.cols + col + 1;
    return (state.edges[top] >= 0) + (state.edges[bottom] >= 0) +
        (state.edges[left] >= 0) + (state.edges[right] >= 0);
}

void apply_move_in_place(State& state, const Board& board, int move) {
    const int player = state.current_player;
    state.edges[move] = static_cast<int8_t>(player);
    state.edge_count += 1;

    int scored = 0;
    const int box_a = board.edge_box_a[move];
    if (box_a >= 0 && state.boxes[box_a] < 0 && box_edge_count(state, board, box_a) == 4) {
        state.boxes[box_a] = static_cast<int8_t>(player);
        state.scores[player] += 1;
        scored += 1;
    }
    const int box_b = board.edge_box_b[move];
    if (box_b >= 0 && state.boxes[box_b] < 0 && box_edge_count(state, board, box_b) == 4) {
        state.boxes[box_b] = static_cast<int8_t>(player);
        state.scores[player] += 1;
        scored += 1;
    }

    if (scored == 0) {
        state.current_player = 1 - state.current_player;
    }
}

double terminal_value(const State& state, const Board& board, int player) {
    if (board.total_boxes <= 0) {
        return 0.0;
    }
    const int opponent = 1 - player;
    return static_cast<double>(state.scores[player] - state.scores[opponent]) /
        static_cast<double>(board.total_boxes);
}

int move_index_for(const Board& board, const std::string& move) {
    auto found = board.move_to_index.find(move);
    if (found == board.move_to_index.end()) {
        throw std::invalid_argument("Unknown edge id for board: " + move);
    }
    return found->second;
}

std::string unicode_to_string(PyObject* value, const char* context) {
    if (!PyUnicode_Check(value)) {
        PyErr_Format(PyExc_ValueError, "expected string for %s", context);
        return "";
    }
    Py_ssize_t size = 0;
    const char* data = PyUnicode_AsUTF8AndSize(value, &size);
    if (data == nullptr) {
        return "";
    }
    return std::string(data, static_cast<size_t>(size));
}

State state_from_snapshot(PyObject* snapshot, const Board& board) {
    State state;
    state.edges.assign(board.action_count, -1);
    state.boxes.assign(board.total_boxes, -1);

    PyObject* current_player = borrowed_dict_item(snapshot, "currentPlayer");
    if (current_player == nullptr) {
        return state;
    }
    state.current_player = static_cast<int>(long_from_object(current_player, "currentPlayer"));
    if (PyErr_Occurred()) {
        return state;
    }

    PyObject* scores = borrowed_dict_item(snapshot, "scores");
    if (scores == nullptr) {
        return state;
    }
    PyObject* scores_fast = PySequence_Fast(scores, "scores must be a sequence");
    if (scores_fast == nullptr) {
        return state;
    }
    if (PySequence_Fast_GET_SIZE(scores_fast) != 2) {
        Py_DECREF(scores_fast);
        PyErr_SetString(PyExc_ValueError, "scores must contain two entries");
        return state;
    }
    state.scores[0] = static_cast<int>(long_from_object(PySequence_Fast_GET_ITEM(scores_fast, 0), "scores[0]"));
    state.scores[1] = static_cast<int>(long_from_object(PySequence_Fast_GET_ITEM(scores_fast, 1), "scores[1]"));
    Py_DECREF(scores_fast);
    if (PyErr_Occurred()) {
        return state;
    }

    PyObject* edges = borrowed_dict_item(snapshot, "edges");
    if (edges == nullptr) {
        return state;
    }
    PyObject* edges_fast = PySequence_Fast(edges, "edges must be a sequence");
    if (edges_fast == nullptr) {
        return state;
    }
    const Py_ssize_t edge_size = PySequence_Fast_GET_SIZE(edges_fast);
    for (Py_ssize_t i = 0; i < edge_size; ++i) {
        std::string edge = unicode_to_string(PySequence_Fast_GET_ITEM(edges_fast, i), "edges");
        if (PyErr_Occurred()) {
            Py_DECREF(edges_fast);
            return state;
        }
        const int move = move_index_for(board, edge);
        if (state.edges[move] < 0) {
            state.edge_count += 1;
        }
        state.edges[move] = 0;
    }
    Py_DECREF(edges_fast);

    PyObject* edge_owners = PyDict_GetItemString(snapshot, "edgeOwners");
    if (edge_owners != nullptr) {
        PyObject* owners_fast = PySequence_Fast(edge_owners, "edgeOwners must be a sequence");
        if (owners_fast == nullptr) {
            return state;
        }
        const Py_ssize_t owner_size = PySequence_Fast_GET_SIZE(owners_fast);
        for (Py_ssize_t i = 0; i < owner_size; ++i) {
            PyObject* entry = PySequence_Fast(PySequence_Fast_GET_ITEM(owners_fast, i), "edgeOwners entries must be sequences");
            if (entry == nullptr) {
                Py_DECREF(owners_fast);
                return state;
            }
            if (PySequence_Fast_GET_SIZE(entry) != 2) {
                Py_DECREF(entry);
                Py_DECREF(owners_fast);
                PyErr_SetString(PyExc_ValueError, "edgeOwners entries must contain edge and owner");
                return state;
            }
            std::string edge = unicode_to_string(PySequence_Fast_GET_ITEM(entry, 0), "edgeOwners edge");
            const int owner = static_cast<int>(long_from_object(PySequence_Fast_GET_ITEM(entry, 1), "edgeOwners owner"));
            Py_DECREF(entry);
            if (PyErr_Occurred()) {
                Py_DECREF(owners_fast);
                return state;
            }
            state.edges[move_index_for(board, edge)] = static_cast<int8_t>(owner);
        }
        Py_DECREF(owners_fast);
    }

    PyObject* boxes = borrowed_dict_item(snapshot, "boxes");
    if (boxes == nullptr) {
        return state;
    }
    PyObject* boxes_fast = PySequence_Fast(boxes, "boxes must be a sequence");
    if (boxes_fast == nullptr) {
        return state;
    }
    const Py_ssize_t box_rows = PySequence_Fast_GET_SIZE(boxes_fast);
    if (box_rows != board.rows - 1) {
        Py_DECREF(boxes_fast);
        PyErr_SetString(PyExc_ValueError, "boxes row count does not match board");
        return state;
    }
    for (Py_ssize_t row = 0; row < box_rows; ++row) {
        PyObject* row_sequence = PySequence_Fast(PySequence_Fast_GET_ITEM(boxes_fast, row), "box rows must be sequences");
        if (row_sequence == nullptr) {
            Py_DECREF(boxes_fast);
            return state;
        }
        if (PySequence_Fast_GET_SIZE(row_sequence) != board.cols - 1) {
            Py_DECREF(row_sequence);
            Py_DECREF(boxes_fast);
            PyErr_SetString(PyExc_ValueError, "boxes column count does not match board");
            return state;
        }
        for (Py_ssize_t col = 0; col < board.cols - 1; ++col) {
            PyObject* owner = PySequence_Fast_GET_ITEM(row_sequence, col);
            const int box_index = static_cast<int>(row) * (board.cols - 1) + static_cast<int>(col);
            if (owner == Py_None) {
                state.boxes[box_index] = -1;
            } else {
                state.boxes[box_index] = static_cast<int8_t>(long_from_object(owner, "box owner"));
                if (PyErr_Occurred()) {
                    Py_DECREF(row_sequence);
                    Py_DECREF(boxes_fast);
                    return state;
                }
            }
        }
        Py_DECREF(row_sequence);
    }
    Py_DECREF(boxes_fast);
    return state;
}

PyObject* snapshot_from_state(const State& state, const Board& board) {
    PyObject* snapshot = PyDict_New();
    if (snapshot == nullptr) {
        return nullptr;
    }
    if (!set_item_steals(snapshot, "rows", PyLong_FromLong(board.rows)) ||
        !set_item_steals(snapshot, "cols", PyLong_FromLong(board.cols)) ||
        !set_item_steals(snapshot, "currentPlayer", PyLong_FromLong(state.current_player))) {
        Py_DECREF(snapshot);
        return nullptr;
    }

    PyObject* edges = PyList_New(0);
    PyObject* edge_owners = PyList_New(0);
    if (edges == nullptr || edge_owners == nullptr) {
        Py_XDECREF(edges);
        Py_XDECREF(edge_owners);
        Py_DECREF(snapshot);
        return nullptr;
    }
    for (int move = 0; move < board.action_count; ++move) {
        if (state.edges[move] < 0) {
            continue;
        }
        PyObject* edge_name = PyUnicode_FromString(board.action_ids[move].c_str());
        if (edge_name == nullptr || PyList_Append(edges, edge_name) < 0) {
            Py_XDECREF(edge_name);
            Py_DECREF(edges);
            Py_DECREF(edge_owners);
            Py_DECREF(snapshot);
            return nullptr;
        }
        PyObject* owner_entry = PyList_New(2);
        if (owner_entry == nullptr) {
            Py_DECREF(edge_name);
            Py_DECREF(edges);
            Py_DECREF(edge_owners);
            Py_DECREF(snapshot);
            return nullptr;
        }
        Py_INCREF(edge_name);
        PyList_SET_ITEM(owner_entry, 0, edge_name);
        PyList_SET_ITEM(owner_entry, 1, PyLong_FromLong(state.edges[move]));
        Py_DECREF(edge_name);
        if (PyErr_Occurred() || PyList_Append(edge_owners, owner_entry) < 0) {
            Py_DECREF(owner_entry);
            Py_DECREF(edges);
            Py_DECREF(edge_owners);
            Py_DECREF(snapshot);
            return nullptr;
        }
        Py_DECREF(owner_entry);
    }
    if (PyDict_SetItemString(snapshot, "edges", edges) < 0 ||
        PyDict_SetItemString(snapshot, "edgeOwners", edge_owners) < 0) {
        Py_DECREF(edges);
        Py_DECREF(edge_owners);
        Py_DECREF(snapshot);
        return nullptr;
    }
    Py_DECREF(edges);
    Py_DECREF(edge_owners);

    PyObject* boxes = PyList_New(board.rows - 1);
    if (boxes == nullptr) {
        Py_DECREF(snapshot);
        return nullptr;
    }
    for (int row = 0; row < board.rows - 1; ++row) {
        PyObject* box_row = PyList_New(board.cols - 1);
        if (box_row == nullptr) {
            Py_DECREF(boxes);
            Py_DECREF(snapshot);
            return nullptr;
        }
        for (int col = 0; col < board.cols - 1; ++col) {
            const int owner = state.boxes[row * (board.cols - 1) + col];
            if (owner < 0) {
                Py_INCREF(Py_None);
                PyList_SET_ITEM(box_row, col, Py_None);
            } else {
                PyList_SET_ITEM(box_row, col, PyLong_FromLong(owner));
            }
        }
        PyList_SET_ITEM(boxes, row, box_row);
    }
    if (PyDict_SetItemString(snapshot, "boxes", boxes) < 0) {
        Py_DECREF(boxes);
        Py_DECREF(snapshot);
        return nullptr;
    }
    Py_DECREF(boxes);

    PyObject* scores = PyList_New(2);
    if (scores == nullptr) {
        Py_DECREF(snapshot);
        return nullptr;
    }
    PyList_SET_ITEM(scores, 0, PyLong_FromLong(state.scores[0]));
    PyList_SET_ITEM(scores, 1, PyLong_FromLong(state.scores[1]));
    if (PyErr_Occurred() || PyDict_SetItemString(snapshot, "scores", scores) < 0) {
        Py_DECREF(scores);
        Py_DECREF(snapshot);
        return nullptr;
    }
    Py_DECREF(scores);

    const bool terminal = state.edge_count >= board.action_count;
    if (!set_item_steals(snapshot, "terminal", PyBool_FromLong(terminal ? 1 : 0))) {
        Py_DECREF(snapshot);
        return nullptr;
    }
    if (terminal) {
        PyObject* winner = nullptr;
        if (state.scores[0] > state.scores[1]) {
            winner = PyLong_FromLong(0);
        } else if (state.scores[1] > state.scores[0]) {
            winner = PyLong_FromLong(1);
        } else {
            winner = PyUnicode_FromString("draw");
        }
        if (!set_item_steals(snapshot, "winner", winner)) {
            Py_DECREF(snapshot);
            return nullptr;
        }
    } else if (!set_item_steals(snapshot, "winner", new_ref(Py_None))) {
        Py_DECREF(snapshot);
        return nullptr;
    }

    return snapshot;
}

std::vector<EvalResult> evaluate_snapshots(
    PyObject* batch_evaluator,
    const std::vector<PyObject*>& snapshots,
    const Board& board
) {
    PyObject* snapshot_list = PyList_New(static_cast<Py_ssize_t>(snapshots.size()));
    if (snapshot_list == nullptr) {
        throw std::runtime_error("failed to allocate snapshot batch");
    }
    for (Py_ssize_t index = 0; index < static_cast<Py_ssize_t>(snapshots.size()); ++index) {
        Py_INCREF(snapshots[static_cast<size_t>(index)]);
        PyList_SET_ITEM(snapshot_list, index, snapshots[static_cast<size_t>(index)]);
    }

    PyObject* result = PyObject_CallFunctionObjArgs(batch_evaluator, snapshot_list, nullptr);
    Py_DECREF(snapshot_list);
    if (result == nullptr) {
        throw std::runtime_error("batch evaluator failed");
    }

    PyObject* result_fast = PySequence_Fast(result, "batch evaluator must return a sequence");
    Py_DECREF(result);
    if (result_fast == nullptr) {
        throw std::runtime_error("batch evaluator returned non-sequence");
    }
    if (PySequence_Fast_GET_SIZE(result_fast) != static_cast<Py_ssize_t>(snapshots.size())) {
        Py_DECREF(result_fast);
        throw std::runtime_error("batch evaluator returned the wrong number of results");
    }

    std::vector<EvalResult> parsed;
    parsed.reserve(snapshots.size());
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(result_fast); ++index) {
        PyObject* item = PySequence_Fast_GET_ITEM(result_fast, index);
        PyObject* item_fast = PySequence_Fast(item, "evaluator result must be (priors, value)");
        if (item_fast == nullptr) {
            Py_DECREF(result_fast);
            throw std::runtime_error("evaluator result was not a pair");
        }
        if (PySequence_Fast_GET_SIZE(item_fast) != 2) {
            Py_DECREF(item_fast);
            Py_DECREF(result_fast);
            throw std::runtime_error("evaluator result must contain priors and value");
        }

        PyObject* priors_obj = PySequence_Fast_GET_ITEM(item_fast, 0);
        PyObject* value_obj = PySequence_Fast_GET_ITEM(item_fast, 1);
        EvalResult eval;
        eval.priors.assign(board.action_count, 0.0);

        if (PyDict_Check(priors_obj)) {
            for (int move = 0; move < board.action_count; ++move) {
                PyObject* prior = PyDict_GetItemString(priors_obj, board.action_ids[move].c_str());
                if (prior != nullptr) {
                    eval.priors[move] = PyFloat_AsDouble(prior);
                    if (PyErr_Occurred()) {
                        Py_DECREF(item_fast);
                        Py_DECREF(result_fast);
                        throw std::runtime_error("prior values must be numeric");
                    }
                }
            }
        } else {
            Py_DECREF(item_fast);
            Py_DECREF(result_fast);
            throw std::runtime_error("evaluator priors must be a dict");
        }

        eval.value = PyFloat_AsDouble(value_obj);
        Py_DECREF(item_fast);
        if (PyErr_Occurred()) {
            Py_DECREF(result_fast);
            throw std::runtime_error("evaluator value must be numeric");
        }
        parsed.push_back(std::move(eval));
    }
    Py_DECREF(result_fast);
    return parsed;
}

void expand_node(
    std::vector<Node>& nodes,
    int node_index,
    const Board& board,
    const EvalResult& eval
) {
    if (nodes[node_index].expanded || nodes[node_index].state.edge_count >= board.action_count) {
        return;
    }

    State base_state = nodes[node_index].state;
    nodes[node_index].children.assign(board.action_count, -1);
    double total_prior = 0.0;
    std::vector<int> legal_moves;
    legal_moves.reserve(board.action_count - base_state.edge_count);

    for (int move = 0; move < board.action_count; ++move) {
        if (base_state.edges[move] >= 0) {
            continue;
        }
        State child_state = base_state;
        apply_move_in_place(child_state, board, move);
        Node child;
        child.state = std::move(child_state);
        child.move = move;
        child.prior = std::max(0.0, eval.priors[move]);
        child.children.assign(board.action_count, -1);
        const int child_index = static_cast<int>(nodes.size());
        nodes.push_back(std::move(child));
        nodes[node_index].children[move] = child_index;
        total_prior += nodes[child_index].prior;
        legal_moves.push_back(move);
    }

    if (!legal_moves.empty()) {
        if (total_prior <= 0.0) {
            const double uniform = 1.0 / static_cast<double>(legal_moves.size());
            for (int move : legal_moves) {
                nodes[nodes[node_index].children[move]].prior = uniform;
            }
        } else {
            for (int move : legal_moves) {
                nodes[nodes[node_index].children[move]].prior /= total_prior;
            }
        }
    }
    nodes[node_index].expanded = true;
}

void add_root_noise(
    std::vector<Node>& nodes,
    int root_index,
    const Board& board,
    double alpha,
    double fraction,
    std::mt19937& rng
) {
    if (alpha <= 0.0 || fraction <= 0.0) {
        return;
    }
    Node& root = nodes[root_index];
    std::vector<int> child_moves;
    for (int move = 0; move < board.action_count; ++move) {
        if (root.children[move] >= 0) {
            child_moves.push_back(move);
        }
    }
    if (child_moves.empty()) {
        return;
    }

    std::gamma_distribution<double> gamma(alpha, 1.0);
    std::vector<double> noise;
    noise.reserve(child_moves.size());
    double total = 0.0;
    for (size_t index = 0; index < child_moves.size(); ++index) {
        const double sample = gamma(rng);
        noise.push_back(sample);
        total += sample;
    }
    if (total <= 0.0) {
        total = static_cast<double>(child_moves.size());
        std::fill(noise.begin(), noise.end(), 1.0);
    }

    for (size_t index = 0; index < child_moves.size(); ++index) {
        Node& child = nodes[root.children[child_moves[index]]];
        const double noise_value = noise[index] / total;
        child.prior = (1.0 - fraction) * child.prior + fraction * noise_value;
    }
}

int select_child(const std::vector<Node>& nodes, int node_index, double c_puct) {
    const Node& node = nodes[node_index];
    const double sqrt_parent = std::sqrt(static_cast<double>(std::max(node.visits, 1)));
    double best_score = -std::numeric_limits<double>::infinity();
    int best_child = -1;

    for (int child_index : node.children) {
        if (child_index < 0) {
            continue;
        }
        const Node& child = nodes[child_index];
        const double q = child.visits == 0 ? 0.0 : child.value_sum / child.visits;
        const double u = c_puct * child.prior * sqrt_parent / (1.0 + child.visits);
        const double score = q + u;
        if (score > best_score) {
            best_score = score;
            best_child = child_index;
        }
    }

    if (best_child < 0) {
        throw std::runtime_error("expanded node had no children");
    }
    return best_child;
}

Reservation reserve_leaf(
    std::vector<Node>& nodes,
    int root_index,
    const Board& board,
    double c_puct,
    double virtual_loss
) {
    Reservation reservation;
    int node_index = root_index;
    nodes[root_index].visits += 1;

    while (nodes[node_index].expanded && nodes[node_index].state.edge_count < board.action_count) {
        const int child_index = select_child(nodes, node_index, c_puct);
        nodes[child_index].visits += 1;
        nodes[child_index].value_sum -= virtual_loss;
        reservation.path.push_back({node_index, child_index});
        node_index = child_index;
    }

    reservation.leaf = node_index;
    reservation.leaf_player = nodes[node_index].state.current_player;
    if (nodes[node_index].state.edge_count >= board.action_count) {
        reservation.terminal = true;
        reservation.terminal_value = terminal_value(
            nodes[node_index].state,
            board,
            reservation.leaf_player
        );
    }
    return reservation;
}

void complete_reservation(
    std::vector<Node>& nodes,
    const Board& board,
    const Reservation& reservation,
    const EvalResult* eval,
    double virtual_loss
) {
    for (const PathEdge& edge : reservation.path) {
        Node& child = nodes[edge.child];
        child.visits -= 1;
        child.value_sum += virtual_loss;
    }

    double leaf_value = reservation.terminal_value;
    if (!reservation.terminal) {
        if (eval == nullptr) {
            throw std::runtime_error("missing evaluator result for nonterminal leaf");
        }
        leaf_value = eval->value;
        expand_node(nodes, reservation.leaf, board, *eval);
    }

    for (const PathEdge& edge : reservation.path) {
        Node& child = nodes[edge.child];
        const Node& parent = nodes[edge.parent];
        const double edge_value = (
            parent.state.current_player == reservation.leaf_player
        ) ? leaf_value : -leaf_value;
        child.visits += 1;
        child.value_sum += edge_value;
    }
}

PyObject* result_from_root(
    const std::vector<Node>& nodes,
    int root_index,
    const Board& board,
    const SearchOptions& options
) {
    const Node& root = nodes[root_index];
    int best_child = -1;
    for (int child_index : root.children) {
        if (child_index < 0) {
            continue;
        }
        const Node& child = nodes[child_index];
        if (best_child < 0) {
            best_child = child_index;
            continue;
        }
        const Node& best = nodes[best_child];
        const double child_mean = child.visits == 0 ? 0.0 : child.value_sum / child.visits;
        const double best_mean = best.visits == 0 ? 0.0 : best.value_sum / best.visits;
        if (child.visits > best.visits ||
            (child.visits == best.visits && child_mean > best_mean) ||
            (child.visits == best.visits && child_mean == best_mean &&
             board.action_ids[child.move] > board.action_ids[best.move])) {
            best_child = child_index;
        }
    }
    if (best_child < 0) {
        PyErr_SetString(PyExc_ValueError, "Search did not expand any legal moves.");
        return nullptr;
    }

    PyObject* result = PyDict_New();
    if (result == nullptr) {
        return nullptr;
    }
    if (!set_item_steals(result, "move", PyUnicode_FromString(board.action_ids[nodes[best_child].move].c_str())) ||
        !set_item_steals(result, "simulations", PyLong_FromLong(options.simulations)) ||
        !set_item_steals(result, "rootPlayer", PyLong_FromLong(root.state.current_player)) ||
        !set_item_steals(result, "batchSize", PyLong_FromLong(options.batch_size)) ||
        !set_item_steals(result, "virtualLoss", PyFloat_FromDouble(options.virtual_loss))) {
        Py_DECREF(result);
        return nullptr;
    }

    std::vector<int> child_indices;
    for (int child_index : root.children) {
        if (child_index >= 0) {
            child_indices.push_back(child_index);
        }
    }
    std::sort(child_indices.begin(), child_indices.end(), [&](int left_index, int right_index) {
        const Node& left = nodes[left_index];
        const Node& right = nodes[right_index];
        if (left.visits != right.visits) {
            return left.visits > right.visits;
        }
        return board.action_ids[left.move] < board.action_ids[right.move];
    });

    PyObject* stats = PyList_New(static_cast<Py_ssize_t>(child_indices.size()));
    if (stats == nullptr) {
        Py_DECREF(result);
        return nullptr;
    }
    for (Py_ssize_t index = 0; index < static_cast<Py_ssize_t>(child_indices.size()); ++index) {
        const Node& child = nodes[child_indices[static_cast<size_t>(index)]];
        const double mean = child.visits == 0 ? 0.0 : child.value_sum / child.visits;
        PyObject* stat = PyDict_New();
        if (stat == nullptr ||
            !set_item_steals(stat, "move", PyUnicode_FromString(board.action_ids[child.move].c_str())) ||
            !set_item_steals(stat, "visits", PyLong_FromLong(child.visits)) ||
            !set_item_steals(stat, "meanValue", PyFloat_FromDouble(mean))) {
            Py_XDECREF(stat);
            Py_DECREF(stats);
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(stats, index, stat);
    }
    if (PyDict_SetItemString(result, "stats", stats) < 0) {
        Py_DECREF(stats);
        Py_DECREF(result);
        return nullptr;
    }
    Py_DECREF(stats);
    return result;
}

SearchOptions parse_options(
    int simulations,
    double c_puct,
    unsigned int seed,
    double root_dirichlet_alpha,
    double root_exploration_fraction,
    int batch_size,
    double virtual_loss
) {
    if (simulations < 1) {
        throw std::invalid_argument("simulations must be at least 1");
    }
    if (c_puct < 0.0) {
        throw std::invalid_argument("c_puct must be non-negative");
    }
    if (batch_size < 1) {
        throw std::invalid_argument("batch_size must be at least 1");
    }
    if (virtual_loss < 0.0) {
        throw std::invalid_argument("virtual_loss must be non-negative");
    }
    SearchOptions options;
    options.simulations = simulations;
    options.c_puct = c_puct;
    options.seed = seed;
    options.root_dirichlet_alpha = root_dirichlet_alpha;
    options.root_exploration_fraction = root_exploration_fraction;
    options.batch_size = batch_size;
    options.virtual_loss = virtual_loss;
    return options;
}

PyObject* search(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* snapshot = nullptr;
    PyObject* batch_evaluator = nullptr;
    int simulations = 1;
    double c_puct = 1.5;
    unsigned int seed = 1;
    double root_dirichlet_alpha = 0.0;
    double root_exploration_fraction = 0.0;
    int batch_size = 1;
    double virtual_loss = 1.0;
    static const char* keywords[] = {
        "snapshot",
        "batch_evaluator",
        "simulations",
        "c_puct",
        "seed",
        "root_dirichlet_alpha",
        "root_exploration_fraction",
        "batch_size",
        "virtual_loss",
        nullptr,
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OO|idIddid",
            const_cast<char**>(keywords),
            &snapshot,
            &batch_evaluator,
            &simulations,
            &c_puct,
            &seed,
            &root_dirichlet_alpha,
            &root_exploration_fraction,
            &batch_size,
            &virtual_loss
        )) {
        return nullptr;
    }
    if (!PyCallable_Check(batch_evaluator)) {
        PyErr_SetString(PyExc_TypeError, "batch_evaluator must be callable");
        return nullptr;
    }
    if (!PyDict_Check(snapshot)) {
        PyErr_SetString(PyExc_TypeError, "snapshot must be a dict");
        return nullptr;
    }

    try {
        SearchOptions options = parse_options(
            simulations,
            c_puct,
            seed,
            root_dirichlet_alpha,
            root_exploration_fraction,
            batch_size,
            virtual_loss
        );

        PyObject* rows_obj = borrowed_dict_item(snapshot, "rows");
        PyObject* cols_obj = borrowed_dict_item(snapshot, "cols");
        if (rows_obj == nullptr || cols_obj == nullptr) {
            return nullptr;
        }
        const int rows = static_cast<int>(long_from_object(rows_obj, "rows"));
        const int cols = static_cast<int>(long_from_object(cols_obj, "cols"));
        if (PyErr_Occurred()) {
            return nullptr;
        }

        Board board = make_board(rows, cols);
        State root_state = state_from_snapshot(snapshot, board);
        if (PyErr_Occurred()) {
            return nullptr;
        }
        if (root_state.edge_count >= board.action_count) {
            PyErr_SetString(PyExc_ValueError, "Cannot search from a terminal state.");
            return nullptr;
        }

        std::vector<Node> nodes;
        nodes.reserve(static_cast<size_t>(simulations + 1) * 8);
        Node root;
        root.state = std::move(root_state);
        root.children.assign(board.action_count, -1);
        nodes.push_back(std::move(root));

        PyObject* root_snapshot = snapshot_from_state(nodes[0].state, board);
        if (root_snapshot == nullptr) {
            return nullptr;
        }
        std::vector<PyObject*> root_batch = {root_snapshot};
        std::vector<EvalResult> root_eval = evaluate_snapshots(batch_evaluator, root_batch, board);
        Py_DECREF(root_snapshot);
        expand_node(nodes, 0, board, root_eval[0]);

        std::mt19937 rng(options.seed);
        add_root_noise(
            nodes,
            0,
            board,
            options.root_dirichlet_alpha,
            options.root_exploration_fraction,
            rng
        );

        int completed = 0;
        while (completed < options.simulations) {
            const int take = std::min(options.batch_size, options.simulations - completed);
            std::vector<Reservation> reservations;
            reservations.reserve(static_cast<size_t>(take));
            std::vector<PyObject*> eval_snapshots;
            eval_snapshots.reserve(static_cast<size_t>(take));

            for (int index = 0; index < take; ++index) {
                Reservation reservation = reserve_leaf(
                    nodes,
                    0,
                    board,
                    options.c_puct,
                    options.virtual_loss
                );
                if (!reservation.terminal) {
                    reservation.eval_index = static_cast<int>(eval_snapshots.size());
                    PyObject* leaf_snapshot = snapshot_from_state(nodes[reservation.leaf].state, board);
                    if (leaf_snapshot == nullptr) {
                        for (PyObject* pending : eval_snapshots) {
                            Py_DECREF(pending);
                        }
                        return nullptr;
                    }
                    eval_snapshots.push_back(leaf_snapshot);
                }
                reservations.push_back(std::move(reservation));
            }

            std::vector<EvalResult> evals;
            if (!eval_snapshots.empty()) {
                evals = evaluate_snapshots(batch_evaluator, eval_snapshots, board);
            }
            for (PyObject* leaf_snapshot : eval_snapshots) {
                Py_DECREF(leaf_snapshot);
            }

            for (const Reservation& reservation : reservations) {
                const EvalResult* eval = nullptr;
                if (!reservation.terminal) {
                    eval = &evals[static_cast<size_t>(reservation.eval_index)];
                }
                complete_reservation(
                    nodes,
                    board,
                    reservation,
                    eval,
                    options.virtual_loss
                );
                completed += 1;
            }
        }

        return result_from_root(nodes, 0, board, options);
    } catch (const std::invalid_argument& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    } catch (const std::exception& error) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, error.what());
        }
        return nullptr;
    }
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-function-type-mismatch"
#endif
PyMethodDef methods[] = {
    {
        "search",
        reinterpret_cast<PyCFunction>(search),
        METH_VARARGS | METH_KEYWORDS,
        "Run the C++ network-guided MCTS core from a Python state snapshot.",
    },
    {nullptr, nullptr, 0, nullptr},
};
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_fast_ez_mcts_cpp",
    "C++ core for network-guided Dots and Boxes MCTS.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__fast_ez_mcts_cpp() {
    return PyModule_Create(&module);
}
