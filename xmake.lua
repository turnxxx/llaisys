add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

if not is_plat("windows") then
    add_cxflags("-fPIC", {force = true})
    add_asflags("-fPIC", {force = true})
end

add_includedirs("include")
add_includedirs(".")
add_includedirs("src")
add_includedirs("thirdparty")
add_includedirs("thirdparty/nlohmann/include")
add_includedirs("thirdparty/nlohmann/include")
-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("enable-log")
    set_default(true)
    set_showmenu(true)
    set_description("Enable LLAISYS logging macros")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

if has_config("enable-log") then
    add_defines("LLAISYS_ENABLE_LOG")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")
    

    on_install(function (target) end)
target_end()



target("llaisys-model")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-device-cpu")
    add_deps("llaisys-core")
    add_deps("llaisys-ops")
    add_deps("llaisys-ops-cpu")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/model/*.cpp")
    add_files("src/model/**/*.cpp")
    add_files("src/layer/**/*.cpp")
    add_files("src/KVcache/*.cpp")
    add_files("src/KVcache/**/*.cpp")
    
    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    if has_config("nv-gpu") then
        -- Ensure CUDA device-link runs for the final shared library.
        add_rules("cuda")
        add_links("cublas", "cublasLt")
        local conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix and #conda_prefix > 0 then
            local conda_lib = path.join(conda_prefix, "lib")
            add_linkdirs(conda_lib)
            add_rpathdirs(conda_lib)
        end
    end
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-device-cpu")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops-cpu")
    -- 注意：不依赖 llaisys-model，直接编译源文件避免链接问题

    set_languages("cxx17")
    set_warnings("all", "error")
    
    -- C API 接口层
    add_files("src/llaisys/*.cc")
    add_files("src/ops/*/op.cpp")
    if has_config("nv-gpu") then
        -- Compile CUDA sources into final shared library to keep
        -- __cudaRegisterLinkedBinary symbols resolved at load time.
        add_files("src/ops/*/nvidia/*.cu")
    end
    
    -- 直接编译模型相关源文件到共享库（确保符号被导出）
    add_files("src/model/*.cpp")
    add_files("src/model/**/*.cpp")
    add_files("src/layer/**/*.cpp")
    add_files("src/KVcache/*.cpp")
    add_files("src/KVcache/**/*.cpp")
    
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()
