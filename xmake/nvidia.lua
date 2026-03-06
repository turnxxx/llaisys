target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    add_rules("cuda")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    add_rules("cuda")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
