getAttribute = idl -> (
    indxs = indices gens idl;
    attribute = "None";
    
    scanPairs(IndxAll, (key, Idxlist) -> (
        if(isSubset(indxs, Idxlist)) then
            attribute = key;
    ));
           
    return attribute;
)


getIndex = inputArray -> (
    conceptRow = inputArray_0;
    attr = inputArray_1;
    idxAbsolute = {};
    idxRelative = {};
    
    for panel in conceptRow do (       
        for idl in primaryDecomposition(panel) do (
            attrIdl = getAttribute(idl);
            if (attr == attrIdl) and (length support idl == 1) then(     
                if (attr == "Number") then(
                    idxAbsolute  = append(idxAbsolute, (((indices idl_0)_0) - min toList(IndxNum)));
                    idxRelative = append(idxRelative, (((indices idl_0)_0) - (min toList(IndxNum)) + 1));
                )
                else if (attr == "Color") then(
                    idxAbsolute  = append(idxAbsolute, (((indices idl_0)_0) - (min toList(IndxColor))));
                    idxRelative = append(idxRelative, (((indices idl_0)_0) - (min toList(IndxColor))));
                )
                else if (attr == "Size") then(
                    idxAbsolute  = append(idxAbsolute, (((indices idl_0)_0) - min toList(IndxSize)));
                    idxRelative = append(idxRelative, ( ((((indices idl_0)_0) - min toList(IndxSize))+2)%6 +1 ));
                );               
            );
        );        
    );                    
    return {idxAbsolute, idxRelative};
)


fNext = inputArray -> (
    idl = inputArray_0;
    Step = inputArray_1;
    generatorList = (first entries gens idl);
    generatorListNew = {};
    
    if Step < 0 then
        direction = -1
    else direction = 1;    
    
    for gen in generatorList do(
        genNew = 1;
        for idx in indices gen do(
            newIdx = idx + Step;
            inSubseq = false;
            scanPairs(IndxSubSequence, (key, Idxlist) -> (
                if isSubset({idx}, Idxlist) then(
                    inSubseq = true;
                    if isSubset({newIdx}, Idxlist) then
                        genNew = genNew * R_newIdx
                    else
                        genNew = genNew * R_(newIdx - (length toList(Idxlist))*direction);
                );
            ));
            
            if inSubseq == false then
                genNew = genNew * R_31;
            
        );
        generatorListNew = append(generatorListNew, genNew);                                        
    );
    return ideal(generatorListNew);
)


hashtableIntersection = tables -> (
    output = new MutableHashTable;
    table1 = tables_0;
    table2 = tables_1;
    for key in toList(set(keys table1) * set(keys table2)) do(
        if set(table1#key) === set(table2#key) then
            output#key = table1#key;
    );        
    return output;
)

intraInvariance = conceptRow -> (
    Jsumpd = primaryDecomposition(conceptRow_0 + conceptRow_1 + conceptRow_2);
    Jinpd = primaryDecomposition(intersect(conceptRow_0, conceptRow_1, conceptRow_2));
    Jsuminpd = toList(set(Jsumpd) * set(Jinpd));
    output = {};

    for i from 0 to ((length Jsuminpd) - 1) do(
        idl = Jsuminpd_i;
        attr = getAttribute(idl);
        if attr != "None" then
            output = append(output, attr);
    );        
    return output;
)


interInvariance = conceptRow -> (
    Jsumpd = primaryDecomposition(conceptRow_0 + conceptRow_1 + conceptRow_2);
    Jinpd = primaryDecomposition(intersect(conceptRow_0, conceptRow_1, conceptRow_2));
    Jdiffpd = set(Jinpd) - set(Jsumpd) ;
    output = new MutableHashTable ;
    for i from 0 to ((length toList(Jdiffpd)) - 1 ) do (
        idl = (toList(Jdiffpd))_i;
        attr = getAttribute(idl);
        if attr != "None" then(
            if (output#?attr) then
                output#attr = append(output#attr, idl)
            else(
                output#attr = {};
                output#attr = append(output#attr, idl);
            );
        );
    );           
    return output;    
)


compositionalInvariance = conceptRow -> (
    output = new MutableHashTable;
    steplist = {-1,-2, 1, 2};

    for Step in steplist do(
        newRow1 = fNext(fNext(conceptRow_0, Step), Step);
	newRow2 = fNext(conceptRow_1, Step);
	newRow3 = conceptRow_2;

	interPd = set(primaryDecomposition(newRow1)) * set(primaryDecomposition(newRow2)) * set(primaryDecomposition(newRow3));
	
	for i when i < length toList(interPd) do(
	    idl = (toList(interPd))_i;
	);
		
        attr = getAttribute(idl);
        if attr != "None" then(
             output#attr = {Step};	      
        );
        

    );
    
    return output;	
)	


binaryOperatorInvariance = conceptRow -> (
    output = new MutableHashTable;
    for attr in {"Number", "Size", "Color"} do(
        idxRow = getIndex(conceptRow, attr);
        idxRelative = idxRow_1;
        if length idxRelative == 3 then (
            if idxRelative_0 + idxRelative_1 == idxRelative_2 then
                output#attr = {1}
            else if idxRelative_0 - idxRelative_1 == idxRelative_2 then
                output#attr = {-1};
        );
    );
                
    return output;
)    


answerSelection = idls -> (
    commonPattern = {0, 0, 0, 0, 0, 0, 0, 0};
    
    for col when col < length idls do(
        row1 = {idls_col_0, idls_col_1, idls_col_2};
        row2 = {idls_col_3, idls_col_4, idls_col_5};
        Pintra12 = set(intraInvariance(row1)) * set(intraInvariance(row2));
        Pinter12 = hashtableIntersection(interInvariance(row1), interInvariance(row2));
        Pcomp12 = hashtableIntersection(compositionalInvariance(row1), compositionalInvariance(row2));
        Pbinary12 = hashtableIntersection(binaryOperatorInvariance(row1), binaryOperatorInvariance(row2));

        commonPatternCol = {};        
        for i from 8 to 15 do (
            if (gens idls_col_i) != 0 then(
		    row3 = {idls_col_6, idls_col_7, idls_col_i};
		    Pintra123 =  Pintra12 * set(intraInvariance(row3));
		    Pinter123 = hashtableIntersection(Pinter12, interInvariance(row3));
		    Pcomp123 = hashtableIntersection(Pcomp12, compositionalInvariance(row3));
		    Pbinary123 = hashtableIntersection(Pbinary12, binaryOperatorInvariance(row3));
		    commonPatternCol = append(commonPatternCol, ((length toList(Pintra123)) + (length keys Pinter123) + (length keys Pcomp123) +  (length keys Pbinary123)));
	    )else(
	    	    commonPatternCol = append(commonPatternCol, 0);
	    );	    
        );
        
        commonPattern = commonPattern + commonPatternCol;
        
    );    

    selectedAnswer = positions(commonPattern, i -> i == max(commonPattern));
    return selectedAnswer;
    
)

R = QQ[x_0..x_68];
IndxNum = set(0..8);
IndxPos = set(9..31);
IndxType = set(32..36);
IndxColor = set(37..46);
IndxSize = set(47..68);


IndxAll = new HashTable from {
    "Number"=>IndxNum,
    "Position"=>IndxPos,
    "Type"=>IndxType,
    "Color"=>IndxColor,
    "Size"=>IndxSize
}


IndxSubSequence = new HashTable from {
    "Number"=>IndxNum,
    "Position1"=>set(9..17),
    "Position2"=>set(18..21),
    "Position3"=>set(22..25),
    "Type"=>IndxType,
    "Color"=>IndxColor,
    "Size1"=>set(47..50),
    "Size2"=>set(51..56),
    "Size3"=>set(57..62),
    "Size4"=>set(63..68)
}


XposList = x_((toList(IndxPos))_0)..x_((toList(IndxPos))_-1);
Ipos = ideal(XposList);
RQ1 = R/Ipos^2;


idealsMtrixQ = {};
idealsMtrix = {};
for i from 3 to ((length commandLine) - 1) do (
        idl = ideal(value commandLine#i);
	idealsMtrixQ = append(idealsMtrixQ, idl);
	idealsMtrix = append(idealsMtrix, substitute(idl, R));
);


commonPos = {};
Ipd = primaryDecomposition(intersect(idealsMtrixQ_0, idealsMtrixQ_1, idealsMtrixQ_2, idealsMtrixQ_3, idealsMtrixQ_4, idealsMtrixQ_5, idealsMtrixQ_6, idealsMtrixQ_7));
if(gens Ipd_0 != 0) then(
	for idl in Ipd do(
		if(length support idl) >= 1 then(
			if(isSubset((indices gens idl), IndxPos)) then (
				commonPos = support idl;
			);
		);
	);
);

panCheck = true;
for i from 0 to 7 do(
    if((length first entries gens idealsMtrixQ_i) < 2) then
         panCheck = false
)

idealsMtrixList = {idealsMtrix};
if(length commonPos > 0 and panCheck) then (
	for pos in commonPos do(
            matrixIn = {};
	    matrixOm = {};

	    for i from 0 to 15 do(
		idlIn = intersect(idealsMtrixQ_i, ideal(pos));
		matrixIn = append(matrixIn, substitute(idlIn, R));

		RQ2 = R/substitute(ideal(pos), R);
		idlOm = trim(substitute(substitute(idealsMtrixQ_i, RQ2), RQ1));
		matrixOm = append(matrixOm, substitute(idlOm, R));
	    );

            idealsMtrixList = append(idealsMtrixList, matrixIn);
            idealsMtrixList = append(idealsMtrixList, matrixOm);
	
	);
);


use R;
Answer = answerSelection(idealsMtrixList);
print(Answer);
