# -*-coding:utf-8-*-
import idaapi
import idc
import pickle
import os
from idaapi import get_func

idaapi.auto_wait()

file_name = os.path.splitext(idc.get_idb_path())[0]
functions = {}
hash_blocks = {}
binary_name = idaapi.get_input_file_path()

binary_file = pickle.load(open(file_name+".addr.dat", "rb"))

func = None
funcfg = None
bblocks = {}

for seg_ea in Segments():
    if idc.get_segm_name(seg_ea) != ".text":
        continue
    for function_ea in Functions(get_segm_start(seg_ea), get_segm_end(seg_ea)):
        function = dict()
        f_name = get_func_name(function_ea)
        function['blocks'] = list()



        funcfg = idaapi.FlowChart(idaapi.get_func(function_ea))
        for bblock in funcfg:
            source_line = []

            sblock = dict()
            sblock['id'] = bblock.id
            dat = {}
            tlines = []
            oprTypes = []
            for head in Heads(bblock.start_ea, bblock.end_ea):
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
                sline = binary_file.get(head, None)
                if sline:
                    source_line.append(sline)
            sblock['src'] = tlines
            sblock['src2line'] = source_line
            function['blocks'].append(sblock)
            source_line = list(set(source_line))
            source_line.sort()
            hash_key = ";".join(source_line)
            hash_blocks[hash_key] = tlines
        functions[f_name] = function

pickle.dump(functions, open("{}.line.dat".format(file_name), "wb"))
pickle.dump(hash_blocks, open("{}.hash.dat".format(file_name), "wb"))
idc.qexit(0)
