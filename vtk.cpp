template<typename T>
static void put(std::fstream &f, const T value) {
    union {
        char buf[sizeof(T)];
        T val;
    } helper;
    helper.val = value;
    std::reverse(helper.buf, helper.buf + sizeof(T));
    f.write(helper.buf, sizeof(T));
}

template <typename real, int time_order, class Problem>
void Solver<real, time_order, Problem>::save(const std::string &prefix) {
    std::string path(prefix + prob.name() + ".avg." + std::to_string(step()) + ".vtk");
    std::fstream f(path, std::ios::out | std::ios::binary);
    if (!f) {
        std::cerr << "Unable to open file `" << path << "' for writing" << std::endl;
        return;
    }

    u_host = u;
    b_host = b;

    f << "# vtk DataFile Version 3.0\n";
    f << "Shallow water solver output - cell avereages\n";
    f << "BINARY\n";
    f << "DATASET RECTILINEAR_GRID\n";
    f << "DIMENSIONS " << M + 1 << " " << N + 1 << " 1";
    f << "\nX_COORDINATES " << M + 1 << " float\n";
    for (int i = 0; i <= M; i++)
        put<float>(f, i * hx);
    f << "\nY_COORDINATES " << N + 1 << " float\n";
    for (int j = 0; j <= N; j++)
        put<float>(f, j * hy);
    f << "\nZ_COORDINATES " << 1 << " float\n";
    put<float>(f, 0);
    f << "\nCELL_DATA " << N * M;

    f << "\nSCALARS h float\nLOOKUP_TABLE default\n";
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= M; i++)
            put<float>(f, u_host.h(i, j).mid());

    f << "\nSCALARS b float\nLOOKUP_TABLE default\n";
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= M; i++)
            put<float>(f, b_host(i, j).mid());

    f << "\nSCALARS zeta float\nLOOKUP_TABLE default\n";
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= M; i++)
            put<float>(f, b_host(i, j).mid() + u_host.h(i, j).mid());

    f << "\nVECTORS v float\n";
        for (int j = 1; j <= N; j++)
            for (int i = 1; i <= M; i++) {
                real h = u_host.h(i, j).mid();
                real vx = u_host.hu(i, j).mid() / h;
                real vy = u_host.hv(i, j).mid() / h;
                put<float>(f, vx);
                put<float>(f, vy);
                put<float>(f, 0);
            }
    f << std::endl;
    f.close();

    path = prefix + prob.name() + ".rec." + std::to_string(step()) + ".vtk";
    f.open(path, std::ios::out | std::ios::binary);
    if (!f) {
        std::cerr << "Unable to open file `" << path << "' for writing" << std::endl;
        return;
    }
    f << "# vtk DataFile Version 3.0\n";
    f << "Shallow water solver output - reconstructed solution\n";
    f << "BINARY\n";
    f << "DATASET UNSTRUCTURED_GRID\n";
    f << "POINTS " << M * N * 4 << " float\n";
    for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++) {
            put<float>(f, i * hx);
            put<float>(f, j * hy);
            put<float>(f, 0);
            put<float>(f, i * hx + hx);
            put<float>(f, j * hy);
            put<float>(f, 0);
            put<float>(f, i * hx + hx);
            put<float>(f, j * hy + hy);
            put<float>(f, 0);
            put<float>(f, i * hx);
            put<float>(f, j * hy + hy);
            put<float>(f, 0);
        }
    int cnt = 0;
    f << "\nCELLS " << N * M << " " << 5 * N * M << "\n";
    for (int i = 0; i < M * N; i++) {
        put<int>(f, 4);
        put<int>(f, cnt); cnt++;
        put<int>(f, cnt); cnt++;
        put<int>(f, cnt); cnt++;
        put<int>(f, cnt); cnt++;
    }
    f << "\nCELL_TYPES " << N * M << "\n";
    for (int i = 0; i < M * N; i++)
        put<int>(f, 9);
    f << "\nCELL_DATA " << N * M;
    f << "\nVECTORS v float\n";
        for (int j = 1; j <= N; j++)
            for (int i = 1; i <= M; i++) {
                real h = u_host.h(i, j).mid();
                real vx = u_host.hu(i, j).mid() / h;
                real vy = u_host.hv(i, j).mid() / h;
                put<float>(f, vx);
                put<float>(f, vy);
                put<float>(f, 0);
            }
    f << "\nPOINT_DATA " << 4 * N * M;
    f << "\nSCALARS h float\nLOOKUP_TABLE default\n";
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= M; i++) {
            put<float>(f, u_host.h(i, j).ll());
            put<float>(f, u_host.h(i, j).lr());
            put<float>(f, u_host.h(i, j).ur());
            put<float>(f, u_host.h(i, j).ul());
        }

    f << "\nSCALARS b float\nLOOKUP_TABLE default\n";
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= M; i++) {
            put<float>(f, b_host(i, j).ll());
            put<float>(f, b_host(i, j).lr());
            put<float>(f, b_host(i, j).ur());
            put<float>(f, b_host(i, j).ul());
        }

    f << "\nSCALARS zeta float\nLOOKUP_TABLE default\n";
    for (int j = 1; j <= N; j++)
        for (int i = 1; i <= M; i++) {
            put<float>(f, b_host(i, j).ll() + u_host.h(i, j).ll());
            put<float>(f, b_host(i, j).lr() + u_host.h(i, j).lr());
            put<float>(f, b_host(i, j).ur() + u_host.h(i, j).ur());
            put<float>(f, b_host(i, j).ul() + u_host.h(i, j).ul());
        }
}
