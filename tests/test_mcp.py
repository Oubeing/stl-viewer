"""MCP 服务回归（无需 GUI / 无需 OpenGL）。

四层验证：

1. 服务层 mcp_service.py —— 工具 schema、业务结果、路径沙箱；
2. 协议层 mcp_stdio.py —— JSON-RPC 分发、握手、通知、错误码；
3. 子进程 stdio —— 真实的 stdin/stdout 字节流；
4. 官方 MCP 客户端 —— 用 mcp.ClientSession 分别连手写版与 SDK 版，
   证明两个入口都符合协议（SDK 未安装时自动跳过这一层）。

运行：python tests/test_mcp.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stl_viewer.mcp_service import (PROMPTS, PROTOCOL_VERSION, RESOURCES,  # noqa: E402
                                    TOOLS, MCPService, ToolError)
from stl_viewer.mcp_stdio import MCPStdioServer  # noqa: E402
from stl_viewer.mesh import Mesh  # noqa: E402
from stl_viewer.png import read_png  # noqa: E402

PASSED, FAILED, SKIPPED = [], [], []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")


def make_root() -> str:
    tmp = tempfile.mkdtemp(prefix="stl_mcp_")
    os.makedirs(os.path.join(tmp, "models"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, "samples", "drill_blank.stl"),
                os.path.join(tmp, "models", "drill_blank.stl"))
    return tmp


# --------------------------------------------------------------------------
# 服务层
# --------------------------------------------------------------------------
def test_service(root: str) -> None:
    svc = MCPService(root)

    check("工具 schema 完整", len(TOOLS) == 15 and all(
        t.get("name") and t.get("description") and
        t.get("inputSchema", {}).get("type") == "object" for t in TOOLS),
        f"{len(TOOLS)} 个工具")
    check("required 字段都存在于 properties", all(
        set(t["inputSchema"]["required"]) <= set(t["inputSchema"]["properties"])
        for t in TOOLS))
    check("资源与提示词已定义", len(RESOURCES) == 4 and len(PROMPTS) == 2)

    info = svc.call_tool("server_info", {}).data
    check("server_info 声明工作区与口径",
          info["workspace_root"] == svc.root and "投影长度" in info["policy"])
    check("所有工具都有实现", all(
        callable(getattr(svc, "_tool_" + t["name"], None)) for t in TOOLS))

    models = svc.call_tool("list_models", {}).data["files"]
    check("list_models 递归找到模型",
          any(f["path"].endswith("drill_blank.stl") for f in models),
          f"{len(models)} 个文件")

    loaded = svc.call_tool("load_model", {"path": "models/drill_blank.stl"})
    check("load_model 载入成功", loaded.data["object"]["triangles"] > 0)
    check("场景状态同步", len(svc.scene.objects) == 1 and
          svc.scene.has_geometry())
    for bad in ("../../Windows/win.ini", "models/missing.stl"):
        try:
            svc.call_tool("load_model", {"path": bad})
            check(f"非法路径被拒绝 {bad}", False)
        except ToolError:
            check(f"非法路径被拒绝 {bad}", True)

    st = svc.call_tool("inspect_model", {}).data["statistics"]
    check("inspect_model 给出拓扑与体积",
          st["triangles"] > 0 and "closed" in st and st["area"] > 0)
    check("list_objects 列出对象",
          len(svc.call_tool("list_objects", {}).data["objects"]) == 1)

    svc.call_tool("set_view", {"view": "top"})
    check("set_view 切到 TOP", abs(svc.camera.pitch - 89.9) < 1e-6)
    try:
        svc.call_tool("set_view", {"view": "no_such"})
        check("非法视角被拒绝", False)
    except ToolError:
        check("非法视角被拒绝", True)

    line = svc.call_tool("measure", {
        "kind": "line3d", "points": [[0, 0, 0], [30, 0, 0]]}).data
    check("长度按投影口径", abs(line["length_mm"] - 30.0) < 1e-6,
          f"{line['length_mm']}")
    check("测量结果带 mode", line["mode"] == "view")
    m3d = svc.call_tool("measure", {
        "kind": "line3d", "points": [[0, 0, 0], [10, 0, 0]],
        "mode": "3d"}).data
    check("3d 模式可切换", m3d["mode"] == "3d")

    circle = svc.call_tool("measure", {
        "kind": "circle3p",
        "points": [[0, 0, 5], [10, 0, 5], [0, 10, 5]]}).data
    check("三点圆是平面正圆",
          abs(circle["radius"] - 5 * 2 ** 0.5) < 1e-6 and
          abs(circle["normal"][2] - 1.0) < 1e-9,
          f"R={circle['radius']:.4f}")
    for bad in ({"kind": "line3d", "points": [[0, 0, 0]]},
                {"kind": "circle3p",
                 "points": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]},
                {"kind": "unknown", "points": [[0, 0, 0]]}):
        try:
            svc.call_tool("measure", bad)
            check(f"非法测量被拒绝 {bad['kind']}", False)
        except ToolError:
            check(f"非法测量被拒绝 {bad['kind']}", True)

    check("测量结果累积",
          len(svc.call_tool("list_measurements", {}).data["items"]) == 3)
    check("clear_measurements 清空",
          svc.call_tool("clear_measurements", {}).data["removed"] == 3)

    pic = svc.call_tool("render_picture", {"width": 120, "height": 90})
    parts = pic.to_content()
    raw = base64.b64decode(parts[1]["data"])
    check("渲染返回 text + image 两种内容",
          len(parts) == 2 and parts[0]["type"] == "text" and
          parts[1]["type"] == "image" and raw.startswith(b"\x89PNG"))
    saved = svc.call_tool("render_picture", {
        "path": "out/shot.png", "width": 100, "height": 80})
    img = read_png(os.path.join(root, "out", "shot.png"))
    check("渲染可写盘且 PNG 可读",
          saved.data["path"] == "out/shot.png" and img.shape == (80, 100, 3))
    try:
        svc.call_tool("render_picture", {"width": 2000, "height": 2000})
        check("超大图片被像素上限拒绝", False)
    except ToolError:
        check("超大图片被像素上限拒绝", True)

    binary = svc.call_tool("export_stl", {"path": "out/part.stl"})
    back = Mesh.load(os.path.join(root, "out", "part.stl"))
    check("导出二进制 STL 可读回",
          binary.data["triangles"] == back.triangle_count)
    ascii_out = svc.call_tool("export_stl", {
        "path": "out/part_ascii.stl", "ascii": True, "weld": True})
    check("导出 ASCII STL", ascii_out.data["format"] == "ascii" and
          os.path.getsize(os.path.join(root, "out", "part_ascii.stl")) > 1000)
    parts = svc.call_tool("split_components", {"out_prefix": "out/part_"})
    check("拆分连通分量", parts.data["components"] >= 1 and
          len(parts.data["files"]) == parts.data["components"])
    radius = svc.call_tool("section_radius", {"z": 10.0}).data
    check("截面回转半径", radius["radius_mm"] > 0)

    for uri, mime in (("stl://scene/objects", "application/json"),
                      ("stl://scene/measurements", "application/json"),
                      ("stl://guide/measurement", "text/markdown"),
                      ("stl://guide/tools", "text/markdown")):
        got_mime, text = svc.read_resource(uri)
        if uri.endswith("measurements"):
            # 上一步 clear_measurements 清空过，空数组也是合法资源内容
            ok = got_mime == mime and json.loads(text) == []
        else:
            ok = got_mime == mime and len(text) > 20
        check(f"资源可读 {uri}", ok)
    prompt = svc.get_prompt("inspect_model", {"path": "models/a.stl"})
    check("提示词返回消息", len(prompt["messages"]) == 1 and
          "models/a.stl" in prompt["messages"][0]["content"]["text"])
    try:
        svc.read_resource("stl://nope")
        check("未知资源被拒绝", False)
    except ToolError:
        check("未知资源被拒绝", True)

    check("remove_object 生效",
          svc.call_tool("remove_object", {"index": 0}).data["objects"] == [])
    check("clear_scene 生效", "已清空" in svc.call_tool("clear_scene", {}).summary)


# --------------------------------------------------------------------------
# 协议层
# --------------------------------------------------------------------------
def test_protocol(root: str) -> None:
    svc = MCPService(root)
    srv = MCPStdioServer(svc, log=io.StringIO())

    def req(mid, method, params=None):
        msg = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params is not None:
            msg["params"] = params
        return srv.handle(msg)

    init = req(1, "initialize", {"protocolVersion": "2025-11-25",
                                 "capabilities": {},
                                 "clientInfo": {"name": "t", "version": "1"}})
    check("initialize 握手", init["result"]["serverInfo"]["name"] == "stl-viewer")
    check("协议版本回显客户端版本",
          init["result"]["protocolVersion"] == "2025-11-25")
    fallback = req(2, "initialize", {"protocolVersion": "1999-01-01"})
    check("未知协议版本降级",
          fallback["result"]["protocolVersion"] == PROTOCOL_VERSION)
    check("capabilities 声明三类原语",
          all(k in init["result"]["capabilities"]
              for k in ("tools", "resources", "prompts")))
    check("通知不产生响应",
          srv.handle({"jsonrpc": "2.0",
                      "method": "notifications/initialized"}) is None)
    check("ping 返回空结果", req(3, "ping")["result"] == {})
    check("tools/list 返回工具", len(req(4, "tools/list")["result"]["tools"]) == 15)

    ok = req(5, "tools/call", {"name": "load_model",
                               "arguments": {"path": "models/drill_blank.stl"}})
    check("tools/call 正常结果",
          ok["result"]["isError"] is False and
          ok["result"]["content"][0]["type"] == "text" and
          "已载入" in ok["result"]["content"][0]["text"])
    bad_tool = req(6, "tools/call", {"name": "no_such", "arguments": {}})
    check("工具错误用 isError 而非 JSON-RPC error",
          bad_tool["result"]["isError"] is True and "error" not in bad_tool)
    bad_path = req(7, "tools/call", {
        "name": "load_model", "arguments": {"path": "../secret.txt"}})
    check("路径沙箱错误回传给 AI",
          bad_path["result"]["isError"] is True and
          "越出工作区" in bad_path["result"]["content"][0]["text"])
    check("未知方法 -32601",
          req(8, "no/such")["error"]["code"] == -32601)
    check("非法 JSON-RPC -32600",
          srv.handle({"id": 9, "method": "ping"})["error"]["code"] == -32600)
    check("resources/list", len(req(10, "resources/list")["result"]["resources"]) == 4)
    check("resources/templates/list",
          req(11, "resources/templates/list")["result"]["resourceTemplates"] == [])
    guide = req(12, "resources/read", {"uri": "stl://guide/measurement"})
    check("resources/read 返回 contents",
          guide["result"]["contents"][0]["mimeType"] == "text/markdown" and
          "投影长度" in guide["result"]["contents"][0]["text"])
    check("prompts/list", len(req(13, "prompts/list")["result"]["prompts"]) == 2)
    pr = req(14, "prompts/get", {"name": "measure_plan",
                                 "arguments": {"goal": "量台阶"}})
    check("prompts/get 返回消息",
          "量台阶" in pr["result"]["messages"][0]["content"]["text"])

    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": PROTOCOL_VERSION}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        "not-json",
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
    ]
    stream = "\n".join(m if isinstance(m, str) else json.dumps(m)
                       for m in messages) + "\n"
    out = io.StringIO()
    srv.serve(stdin=io.StringIO(stream), stdout=out)
    lines = [json.loads(x) for x in out.getvalue().splitlines() if x.strip()]
    check("stdio 一行一响应、通知不回复", len(lines) == 4, f"{len(lines)} 行")
    check("解析错误 -32700", lines[2]["error"]["code"] == -32700)
    check("响应 id 与请求一致",
          [x["id"] for x in lines] == [1, 2, None, 3])


# --------------------------------------------------------------------------
# 真实子进程 stdio
# --------------------------------------------------------------------------
def test_subprocess(root: str) -> None:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": ROOT}
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                    "clientInfo": {"name": "t", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "load_model",
                    "arguments": {"path": "models/drill_blank.stl"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "measure",
                    "arguments": {"kind": "line3d",
                                  "points": [[0, 0, 0], [30, 0, 0]]}}},
        {"jsonrpc": "2.0", "id": 5, "method": "resources/read",
         "params": {"uri": "stl://scene/objects"}},
    ]
    payload = ("\n".join(json.dumps(r, ensure_ascii=False) for r in reqs)
               + "\n").encode("utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "stl_viewer.mcp_stdio", "--root", root],
        input=payload, capture_output=True, env=env, cwd=ROOT, timeout=90)
    check("子进程正常退出", proc.returncode == 0, f"rc={proc.returncode}")
    lines = [x for x in proc.stdout.decode("utf-8").splitlines() if x.strip()]
    parsed = [json.loads(x) for x in lines]
    check("stdout 每行都是合法协议消息", len(parsed) == 5 and
          all(p.get("jsonrpc") == "2.0" for p in parsed))
    check("子进程内工具真的执行了",
          "已载入" in parsed[2]["result"]["content"][0]["text"] and
          "L=30.000" in parsed[3]["result"]["content"][0]["text"])
    check("日志只进 stderr，未污染 stdout",
          b"stdio ready" in proc.stderr and b"{" not in proc.stderr.splitlines()[0])
    check("资源内容经子进程读回",
          "drill_blank" in parsed[4]["result"]["contents"][0]["text"])


# --------------------------------------------------------------------------
# 官方 MCP 客户端（可选）
# --------------------------------------------------------------------------
async def _client_session(target: str, root: str) -> dict:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": ROOT}
    params = StdioServerParameters(
        command=sys.executable, args=["-m", target, "--root", root],
        cwd=ROOT, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            loaded = await session.call_tool(
                "load_model", {"path": "models/drill_blank.stl"})
            measured = await session.call_tool(
                "measure", {"kind": "line3d", "points": [[0, 0, 0], [30, 0, 0]]})
            picture = await session.call_tool(
                "render_picture", {"width": 120, "height": 90})
            resources = await session.list_resources()
            guide = await session.read_resource("stl://guide/measurement")
            prompts = await session.list_prompts()
            prompt = await session.get_prompt(
                "inspect_model", {"path": "models/drill_blank.stl"})
            return {
                "server": init.server_info.name,
                "tools": len(tools.tools),
                "load_error": loaded.is_error,
                "measure_text": measured.content[0].text,
                "content_types": [c.type for c in picture.content],
                "resources": len(resources.resources),
                "guide": guide.contents[0].text,
                "prompts": len(prompts.prompts),
                "prompt_messages": len(prompt.messages),
            }


def test_official_client(root: str) -> None:
    try:
        import mcp  # noqa: F401
    except ImportError:
        SKIPPED.append("官方 SDK 客户端测试")
        print("  [SKIP] 未安装 mcp SDK，跳过第 4 层（pip install \"mcp>=1.2\"）")
        return
    for target, label in (("stl_viewer.mcp_stdio", "手写 stdio 版"),
                          ("stl_viewer.mcp_fastmcp", "官方 SDK 版")):
        try:
            r = asyncio.run(_client_session(target, root))
        except Exception as exc:                     # pragma: no cover
            check(f"{label} 通过官方客户端端到端", False, f"{type(exc).__name__}: {exc}")
            continue
        check(f"{label} 握手与工具清单",
              r["server"] == "stl-viewer" and r["tools"] == 15)
        check(f"{label} 工具执行与测量",
              r["load_error"] is False and "L=30.000" in r["measure_text"])
        check(f"{label} 返回图片内容",
              r["content_types"] == ["text", "image"])
        check(f"{label} 资源与提示词",
              r["resources"] == 4 and r["prompts"] == 2 and
              r["prompt_messages"] == 1 and "投影长度" in r["guide"])


def main() -> int:
    root = make_root()
    try:
        print("\n=== 服务层 ===")
        test_service(root)
        print("\n=== 协议层 ===")
        test_protocol(root)
        print("\n=== 子进程 stdio ===")
        test_subprocess(root)
        print("\n=== 官方 MCP 客户端 ===")
        test_official_client(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n总计: {len(PASSED)} 通过, {len(FAILED)} 失败, {len(SKIPPED)} 跳过")
    if FAILED:
        print("失败项: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
