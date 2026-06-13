#!/bin/bash

objdump -j .text -D bin/is.S.x.sampling.m5.pdev > is.S.sampling.objdump
objdump -j .text -D bin/is.D.x.sampling.m5.pdev > is.D.sampling.objdump
objdump -j .text -D bin/cg.S.x.sampling.m5.pdev > cg.S.sampling.objdump
objdump -j .text -D bin/cg.E.x.sampling.m5.pdev > cg.E.sampling.objdump
objdump -j .text -D bin/ua.S.x.sampling.m5.pdev > ua.S.sampling.objdump
objdump -j .text -D bin/ua.D.x.sampling.m5.pdev > ua.D.sampling.objdump
