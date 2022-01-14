# -*-coding:utf-8-*-

import idaapi
import idc
import pickle
import os
import idautils
from idaapi import get_func
import ida_nalt
import decimal
import json

idaapi.auto_wait()

cpu_ins_list = GetInstructionList()
cpu_ins_list.sort()

# Downloaded from http://www.logarithmic.net/pfh-files/blog/01208083168/sort.py

"""

   Tarjan's algorithm and topological sorting implementation in Python

   by Paul Harrison

   Public domain, do with it as you will

"""
# Modify Function Names
for seg_ea in Segments():
    if idc.get_segm_name(seg_ea) != ".text":
        continue
    for function_ea in Functions(get_segm_start(seg_ea), get_segm_end(seg_ea)):
        f_name = get_func_name(function_ea)
        if "_Hash_" in f_name:
            continue
        idaapi.set_name(function_ea, f_name+"_Hash_"+str(hash(f_name) % 9973).rjust(4, "0"), idaapi.SN_FORCE)


def strongly_connected_components(graph):
    """ Find the strongly connected components in a graph using
        Tarjan's algorithm.

        graph should be a dictionary mapping node names to
        lists of successor nodes.
        """

    result = []
    stack = []
    low = {}

    def visit(node):
        if node in low: return
        if node not in graph: graph[node] = []

        num = len(low)
        low[node] = num
        stack_pos = len(stack)
        stack.append(node)

        for successor in graph[node]:
            visit(successor)
            low[node] = min(low[node], low[successor])

        if num == low[node]:
            component = tuple(stack[stack_pos:])
            del stack[stack_pos:]
            result.append(component)
            for item in component:
                low[item] = len(graph)

    for node in dict(graph):
        visit(node)

    return result


def topological_sort(graph):
    count = {}
    for node in graph:
        count[node] = 0
    for node in graph:
        for successor in graph[node]:
            count[successor] += 1

    ready = [node for node in graph if count[node] == 0]

    result = []
    while ready:
        node = ready.pop(-1)
        result.append(node)

        for successor in graph[node]:
            count[successor] -= 1
            if count[successor] == 0:
                ready.append(successor)

    return result


def robust_topological_sort(graph):
    """ First identify strongly connected components,
        then perform a topological sort on these components. """

    components = strongly_connected_components(graph)

    node_component = {}
    for component in components:
        for node in component:
            node_component[node] = component

    component_graph = {}
    for component in components:
        component_graph[component] = []

    for node in graph:
        node_c = node_component[node]
        for successor in graph[node]:
            successor_c = node_component[successor]
            if node_c != successor_c:
                component_graph[node_c].append(successor_c)

    return topological_sort(component_graph)


def primesbelow(N):
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    # """ Input N>=6, Returns a list of primes, 2 <= p < N """
    correction = N % 6 > 1
    N = {0: N, 1: N - 1, 2: N + 4, 3: N + 3, 4: N + 2, 5: N + 1}[N % 6]
    sieve = [True] * (N // 3)
    sieve[0] = False
    for i in range(int(N ** .5) // 3 + 1):
        if sieve[i]:
            k = (3 * i + 1) | 1
            sieve[k * k // 3::2 * k] = [False] * ((N // 6 - (k * k) // 6 - 1) // k + 1)
            sieve[(k * k + 4 * k - 2 * k * (i % 2)) // 3::2 * k] = [False] * (
                        (N // 6 - (k * k + 4 * k - 2 * k * (i % 2)) // 6 - 1) // k + 1)
    return [2, 3] + [(3 * i + 1) | 1 for i in range(1, N // 3 - correction) if sieve[i]]


def find_imported_funcs():
    imports = []
    nimps = ida_nalt.get_import_module_qty()
    print("Found %d import(s)..." % nimps)
    for i in range(nimps):
        name = ida_nalt.get_import_module_name(i)
        if not name:
            print("Failed to get import module name for #%d" % i)
            name = "<unnamed>"

        print("Walking imports for module %s" % name)

        def imp_cb(ea, name, ordinal):
            if not name:
                print("%08x: ordinal #%d" % (ea, ordinal))
                name = ""
            else:
                print("%08x: %s (ordinal #%d)" % (ea, name, ordinal))
            # True -> Continue enumeration
            # False -> Stop enumeration
            imports.append([ea, name, ordinal])
            return True

        ida_nalt.enum_import_names(i, imp_cb)
    print("All done...")
    return imports


def find_import_ref():
    imports = find_imported_funcs()
    R = dict()
    for i, (ea, name, _) in enumerate(imports):
        for xref in idautils.XrefsTo(ea):
            ea = xref.frm
            f = ida_funcs.get_func(ea)
            if f and (f.flags & ida_funcs.FUNC_THUNK) != 0:
                imports.append([f.start_ea, ida_funcs.get_func_name(f.start_ea), 0])
                continue


            if i not in R:
                R[i] = []

            if text_start <= ea <= text_end:
                R[i].append(ea)
            else:
                continue

    return (imports, R)


file_name = os.path.splitext(idc.get_idb_path())[0]
functions = []
binary_name = idaapi.get_input_file_path()
callgraphs = {}
primes = primesbelow(30000)
function_mnemonics_spp = {}
func_type_map = {}
func_type_dict = {}
for seg_ea in Segments():
    if idc.get_segm_name(seg_ea) != ".text":
        continue
    text_start, text_end = get_segm_start(seg_ea), get_segm_end(seg_ea)
    for function_ea in Functions(text_start, text_end):
        function = dict()

        f_name = get_func_name(function_ea)
        func_type = idc.get_type(function_ea)
        if func_type:
            func_type = func_type.replace("__cdecl", "__fastcall")
            func_type_dict[f_name] = func_type
            func_type_map[func_type] = func_type_map.get(func_type, [])
            func_type_map[func_type].append(f_name)

        callgraphs[f_name] = {"caller": [],
                              "callee": []
                              }
        func_caller = []
        for xref in list(XrefsTo(function_ea)):
            caller = xref.frm
            caller_name = idc.get_func_name(caller)
            if caller_name:
                if not text_start <= caller <= text_end:
                    print(caller_name, "Not caller")
                    continue
                func_caller.append(caller_name)
        callgraphs[f_name]["caller"] = func_caller

        function['name'] = f_name
        function['blocks'] = list()
        function['filename'] = binary_name

        constants = []
        nodes = 0
        edges = 0
        size = 0
        instructions = 0
        mnemonics_spp = 1

        funcfg = idaapi.FlowChart(idaapi.get_func(function_ea))
        for bblock in funcfg:
            sblock = dict()
            sblock['id'] = bblock.id
            dat = {}
            tlines = []
            oprTypes = []
            s = get_bytes(bblock.start_ea, bblock.end_ea - bblock.start_ea)

            nodes += 1

            if s is not None:
                sblock['bytes'] = "".join("{:02x}".format(c) for c in s)
            else:
                print(function['name'])
            for head in Heads(bblock.start_ea, bblock.end_ea):
                mnem = print_insn_mnem(head)
                if mnem in cpu_ins_list:
                    mnemonics_spp *= primes[cpu_ins_list.index(mnem)]
                size += get_item_size(head)
                instructions += 1

                tline = list()
                oprType = list()
                mnem = idc.print_insn_mnem(head)
                if mnem == "":
                    continue
                mnem = idc.GetDisasm(head).split()[0]
                tline.append(mnem)
                for i in range(5):
                    opd = idc.print_operand(head, i)
                    tp = idc.get_operand_type(head, i)
                    if opd == "" or tp is None:
                        continue
                    tline.append(opd)
                    oprType.append(tp)
                tlines.append(tline)
                oprTypes.append(oprType)

                refdata = list(DataRefsFrom(head))
                if len(refdata) > 0:
                    for ref in refdata:
                        dat[head] = format(get_qword(ref), 'x')[::-1]

                drefs = list(DataRefsFrom(head))
                if len(drefs) > 0:
                    for dref in drefs:
                        if get_func(dref) is None:
                            str_constant = get_strlit_contents(dref, -1, -1)
                            if str_constant is not None:
                                str_constant = str_constant.decode("utf-8", "backslashreplace")
                                if str_constant not in constants:
                                    constants.append(str_constant)

            sblock['src'] = tlines
            sblock['oprType'] = oprTypes
            sblock['dat'] = dat

            succs = list()
            for succ_block in bblock.succs():
                edges += 1
                succs.append(succ_block.id)
            sblock['succs'] = succs
            function['blocks'].append(sblock)
        function['constants'] = constants
        function['size'] = size
        function['nodes'] = nodes
        function['edges'] = edges
        function['instructions'] = instructions
        function['mnemonics_spp'] = mnemonics_spp
        functions.append(function)
        function_mnemonics_spp[str(mnemonics_spp)] = (instructions, f_name)

    for cur_func in callgraphs:
        for caller in callgraphs[cur_func]['caller']:

            try:
                callgraphs[caller]['callee'].append(cur_func)
            except:
                print("{}->{}".format(caller, cur_func))
                print(callgraphs[cur_func]['caller'])

callgraph_cache = {}


def calculate_md_index(input_func_list, callgraphs, k, isCaller=True):
    global callgraph_cache
    cur_k = k - 1
    tag = 'caller' if isCaller else 'callee'
    ret = {}
    related_funcs = []
    traversed = []
    for func in input_func_list:
        if func in traversed:
            continue
        else:
            traversed.append(func)
        tmp = list(set(callgraphs[func][tag]))
        tmp.sort()
        ret[func] = tmp
        related_funcs.extend(list(set(callgraphs[func][tag])))
        if cur_k > 0:
            if k == 999:
                callgraph_cache[func] = callgraph_cache.get(func, {"caller": None, "callee": None})
                if callgraph_cache[func][tag]:
                    ans, sub_funcs = callgraph_cache[func][tag]
                else:
                    ans, sub_funcs = calculate_md_index(callgraphs[func][tag], callgraphs, cur_k, isCaller=isCaller)
                    ret.update(ans)
                    callgraph_cache[func][tag] = (ans, sub_funcs)
            else:
                ans, sub_funcs = calculate_md_index(callgraphs[func][tag], callgraphs, cur_k, isCaller=isCaller)
                ret.update(ans)
            related_funcs.extend(sub_funcs)
    return ret, related_funcs


def callgraph_index(isCaller=True):
    fcg_md_index = {}
    fcg_md_map_func = {}
    k = 8
    count_N = 0
    for func in callgraphs:
        print(func, count_N, "/", len(callgraphs))
        count_N += 1
        cg, all_funcs = calculate_md_index([func], callgraphs, k, isCaller=isCaller)
        tmp_cg = {}

        all_funcs.append(func)
        all_funcs = list(set(all_funcs))
        all_funcs.sort()

        for f in all_funcs:
            tmp_cg[f] = {}
            tmp_cg[f]['caller'] = len(set([_ for _ in callgraphs[f]['caller'] if _ in all_funcs]))
            tmp_cg[f]['callee'] = len(set([_ for _ in callgraphs[f]['callee'] if _ in all_funcs]))

        func_map = {}
        n = 0
        for _ in all_funcs:
            func_map[_] = n
            n += 1
        cg_topological = {}
        for f in cg:
            cg_topological[func_map[f]] = [func_map[_] for _ in cg[f]]

        try:
            cg_topological_sorted = robust_topological_sort(cg_topological)
            cg_topological = json.dumps(cg_topological_sorted)
        except:
            cg_topological = None
        if cg_topological:
            cg_topo_order = {}
            for i, scc in enumerate(cg_topological_sorted):
                for f in scc:
                    cg_topo_order[f] = i
            tuples = []
            for src in cg:
                for dst in cg[src]:
                    src_caller = len(set(callgraphs[src]['caller']))
                    src_callee = len(set(callgraphs[src]['callee']))
                    dst_caller = len(set(callgraphs[dst]['caller']))
                    dst_callee = len(set(callgraphs[dst]['callee']))
                    if src == "main":
                        src_callee = 53
                    if dst == "main":
                        dst_callee = 53
                    tuples.append((cg_topo_order[func_map[src]],
                                   src_caller,
                                   src_callee,
                                   dst_caller,
                                   dst_callee,))


            rt2, rt3, rt5, rt7 = (decimal.Decimal(p).sqrt() for p in (2, 3, 5, 7))
            emb_tuples = (sum((z0, z1 * rt2, z2 * rt3, z3 * rt5, z4 * rt7))
                          for z0, z1, z2, z3, z4 in tuples)
            md_index = sum((1 / emb_t.sqrt() for emb_t in emb_tuples))
            md_index = md_index

            fcg_md_index[func] = md_index
        else:
            fcg_md_index[func] = 0

    return fcg_md_index


caller_fcg = callgraph_index(isCaller=True)
callee_fcg = callgraph_index(isCaller=False)
weight1 = decimal.Decimal(17).sqrt()
weight2 = decimal.Decimal(13).sqrt()
fcg_md_index = {}
fcg_md_map_func = {}
for func in caller_fcg:
    md_index = caller_fcg[func] * weight1 + callee_fcg[func] * weight2
    if md_index == 0:
        md_index = 0
    md_index = str(md_index)
    fcg_md_index[func] = md_index
    fcg_md_map_func[md_index] = fcg_md_map_func.get(md_index, [])
    fcg_md_map_func[md_index].append(func)
for md_index in fcg_md_map_func:
    fcg_md_map_func[md_index]
fcg_md_index_class = {'map': fcg_md_map_func, 'func': fcg_md_index}
pickle.dump(fcg_md_index_class, open("{}.fcg.dat".format(file_name), "wb"))

STRINGS = idautils.Strings(False)
STRINGS_DICT = {}
STRINGS_LIST = {}
for s in STRINGS:
    _key = str(s)
    address = s.ea
    funcs = []
    for ea in DataRefsTo(s.ea):
        function_name = get_func_name(ea)
        if function_name:
            funcs.append(function_name)
    if funcs:
        STRINGS_LIST[_key] = (s.ea, s.length, len(set(funcs)))
        STRINGS_DICT[_key] = funcs

imports, R = find_import_ref()
imports_map = {}
function_map = {}
for k, v in R.items():
    if v:
        for ea in v:
            import_func = imports[k][1]
            func = idaapi.get_func_name(ea)
            imports_list = imports_map.get(import_func, [])
            imports_list.append(func)
            imports_map[import_func] = imports_list
            function_list = function_map.get(func, [])
            function_list.append(import_func)
            function_map[func] = function_list

pickle.dump((callgraphs, STRINGS_DICT, STRINGS_LIST, imports_map, function_map, func_type_map, func_type_dict),
            open("{}.cmp.dat".format(file_name), "wb"))
pickle.dump(functions, open("{}.pkl".format(file_name), "wb"))
idc.qexit(0)