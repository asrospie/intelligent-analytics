??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02unknown8??
?
hidden_layer_0_1/kernelVarHandleOp*(
shared_namehidden_layer_0_1/kernel*
shape
:*
dtype0*
_output_shapes
: 
?
+hidden_layer_0_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_0_1/kernel*
_output_shapes

:*
dtype0
?
hidden_layer_0_1/biasVarHandleOp*
dtype0*
shape:*&
shared_namehidden_layer_0_1/bias*
_output_shapes
: 
{
)hidden_layer_0_1/bias/Read/ReadVariableOpReadVariableOphidden_layer_0_1/bias*
_output_shapes
:*
dtype0
?
hidden_layer_1_1/kernelVarHandleOp*(
shared_namehidden_layer_1_1/kernel*
shape
:*
_output_shapes
: *
dtype0
?
+hidden_layer_1_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_1_1/kernel*
dtype0*
_output_shapes

:
?
hidden_layer_1_1/biasVarHandleOp*
_output_shapes
: *
shape:*&
shared_namehidden_layer_1_1/bias*
dtype0
{
)hidden_layer_1_1/bias/Read/ReadVariableOpReadVariableOphidden_layer_1_1/bias*
_output_shapes
:*
dtype0
?
hidden_layer_2_1/kernelVarHandleOp*
shape
:*
dtype0*
_output_shapes
: *(
shared_namehidden_layer_2_1/kernel
?
+hidden_layer_2_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_2_1/kernel*
dtype0*
_output_shapes

:
?
hidden_layer_2_1/biasVarHandleOp*
_output_shapes
: *
shape:*
dtype0*&
shared_namehidden_layer_2_1/bias
{
)hidden_layer_2_1/bias/Read/ReadVariableOpReadVariableOphidden_layer_2_1/bias*
_output_shapes
:*
dtype0
?
hidden_layer_3_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *(
shared_namehidden_layer_3_1/kernel*
shape
:
?
+hidden_layer_3_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_3_1/kernel*
dtype0*
_output_shapes

:
?
hidden_layer_3_1/biasVarHandleOp*
_output_shapes
: *&
shared_namehidden_layer_3_1/bias*
dtype0*
shape:
{
)hidden_layer_3_1/bias/Read/ReadVariableOpReadVariableOphidden_layer_3_1/bias*
dtype0*
_output_shapes
:
?
output_layer_1/kernelVarHandleOp*
_output_shapes
: *&
shared_nameoutput_layer_1/kernel*
shape
:*
dtype0

)output_layer_1/kernel/Read/ReadVariableOpReadVariableOpoutput_layer_1/kernel*
dtype0*
_output_shapes

:
~
output_layer_1/biasVarHandleOp*
shape:*
_output_shapes
: *
dtype0*$
shared_nameoutput_layer_1/bias
w
'output_layer_1/bias/Read/ReadVariableOpReadVariableOpoutput_layer_1/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
shared_name	Adam/iter*
dtype0	*
_output_shapes
: *
shape: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
shape: *
dtype0*
shared_nameAdam/beta_1*
_output_shapes
: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
shape: *
shared_nameAdam/beta_2*
_output_shapes
: *
dtype0
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
shape: *
shared_name
Adam/decay*
dtype0*
_output_shapes
: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shared_nametotal*
_output_shapes
: *
dtype0*
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
shared_namecount*
shape: *
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/hidden_layer_0_1/kernel/mVarHandleOp*/
shared_name Adam/hidden_layer_0_1/kernel/m*
_output_shapes
: *
dtype0*
shape
:
?
2Adam/hidden_layer_0_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_0_1/kernel/m*
_output_shapes

:*
dtype0
?
Adam/hidden_layer_0_1/bias/mVarHandleOp*-
shared_nameAdam/hidden_layer_0_1/bias/m*
shape:*
dtype0*
_output_shapes
: 
?
0Adam/hidden_layer_0_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_0_1/bias/m*
dtype0*
_output_shapes
:
?
Adam/hidden_layer_1_1/kernel/mVarHandleOp*
shape
:*
dtype0*
_output_shapes
: */
shared_name Adam/hidden_layer_1_1/kernel/m
?
2Adam/hidden_layer_1_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_1_1/kernel/m*
dtype0*
_output_shapes

:
?
Adam/hidden_layer_1_1/bias/mVarHandleOp*
shape:*-
shared_nameAdam/hidden_layer_1_1/bias/m*
_output_shapes
: *
dtype0
?
0Adam/hidden_layer_1_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_1_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/hidden_layer_2_1/kernel/mVarHandleOp*
_output_shapes
: *
shape
:*
dtype0*/
shared_name Adam/hidden_layer_2_1/kernel/m
?
2Adam/hidden_layer_2_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_2_1/kernel/m*
_output_shapes

:*
dtype0
?
Adam/hidden_layer_2_1/bias/mVarHandleOp*-
shared_nameAdam/hidden_layer_2_1/bias/m*
dtype0*
shape:*
_output_shapes
: 
?
0Adam/hidden_layer_2_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_2_1/bias/m*
dtype0*
_output_shapes
:
?
Adam/hidden_layer_3_1/kernel/mVarHandleOp*
dtype0*/
shared_name Adam/hidden_layer_3_1/kernel/m*
_output_shapes
: *
shape
:
?
2Adam/hidden_layer_3_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_3_1/kernel/m*
dtype0*
_output_shapes

:
?
Adam/hidden_layer_3_1/bias/mVarHandleOp*-
shared_nameAdam/hidden_layer_3_1/bias/m*
shape:*
dtype0*
_output_shapes
: 
?
0Adam/hidden_layer_3_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_3_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/output_layer_1/kernel/mVarHandleOp*-
shared_nameAdam/output_layer_1/kernel/m*
_output_shapes
: *
shape
:*
dtype0
?
0Adam/output_layer_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_layer_1/kernel/m*
_output_shapes

:*
dtype0
?
Adam/output_layer_1/bias/mVarHandleOp*+
shared_nameAdam/output_layer_1/bias/m*
dtype0*
shape:*
_output_shapes
: 
?
.Adam/output_layer_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_layer_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/hidden_layer_0_1/kernel/vVarHandleOp*/
shared_name Adam/hidden_layer_0_1/kernel/v*
_output_shapes
: *
dtype0*
shape
:
?
2Adam/hidden_layer_0_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_0_1/kernel/v*
dtype0*
_output_shapes

:
?
Adam/hidden_layer_0_1/bias/vVarHandleOp*
dtype0*-
shared_nameAdam/hidden_layer_0_1/bias/v*
shape:*
_output_shapes
: 
?
0Adam/hidden_layer_0_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_0_1/bias/v*
dtype0*
_output_shapes
:
?
Adam/hidden_layer_1_1/kernel/vVarHandleOp*/
shared_name Adam/hidden_layer_1_1/kernel/v*
shape
:*
dtype0*
_output_shapes
: 
?
2Adam/hidden_layer_1_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_1_1/kernel/v*
dtype0*
_output_shapes

:
?
Adam/hidden_layer_1_1/bias/vVarHandleOp*
_output_shapes
: *
shape:*
dtype0*-
shared_nameAdam/hidden_layer_1_1/bias/v
?
0Adam/hidden_layer_1_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_1_1/bias/v*
dtype0*
_output_shapes
:
?
Adam/hidden_layer_2_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*/
shared_name Adam/hidden_layer_2_1/kernel/v*
shape
:
?
2Adam/hidden_layer_2_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_2_1/kernel/v*
_output_shapes

:*
dtype0
?
Adam/hidden_layer_2_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*-
shared_nameAdam/hidden_layer_2_1/bias/v*
shape:
?
0Adam/hidden_layer_2_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_2_1/bias/v*
dtype0*
_output_shapes
:
?
Adam/hidden_layer_3_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name Adam/hidden_layer_3_1/kernel/v
?
2Adam/hidden_layer_3_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_3_1/kernel/v*
_output_shapes

:*
dtype0
?
Adam/hidden_layer_3_1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*-
shared_nameAdam/hidden_layer_3_1/bias/v
?
0Adam/hidden_layer_3_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_3_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/output_layer_1/kernel/vVarHandleOp*-
shared_nameAdam/output_layer_1/kernel/v*
_output_shapes
: *
shape
:*
dtype0
?
0Adam/output_layer_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_layer_1/kernel/v*
_output_shapes

:*
dtype0
?
Adam/output_layer_1/bias/vVarHandleOp*
_output_shapes
: *
shape:*+
shared_nameAdam/output_layer_1/bias/v*
dtype0
?
.Adam/output_layer_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_layer_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?9
ConstConst"/device:CPU:0*
dtype0*?9
value?8B?8 B?8
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
h

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
h

)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
?
/iter

0beta_1

1beta_2
	2decay
3learning_ratem\m]m^m_m`ma#mb$mc)md*mevfvgvhvivjvk#vl$vm)vn*vo
F
0
1
2
3
4
5
#6
$7
)8
*9
F
0
1
2
3
4
5
#6
$7
)8
*9
 
?

4layers
	variables
5layer_regularization_losses
6metrics
7non_trainable_variables
	trainable_variables

regularization_losses
 
 
 
 
?

8layers
	variables
9layer_regularization_losses
:metrics
regularization_losses
;non_trainable_variables
trainable_variables
ca
VARIABLE_VALUEhidden_layer_0_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEhidden_layer_0_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

<layers
	variables
=layer_regularization_losses
>metrics
regularization_losses
?non_trainable_variables
trainable_variables
ca
VARIABLE_VALUEhidden_layer_1_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEhidden_layer_1_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

@layers
	variables
Alayer_regularization_losses
Bmetrics
regularization_losses
Cnon_trainable_variables
trainable_variables
ca
VARIABLE_VALUEhidden_layer_2_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEhidden_layer_2_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

Dlayers
	variables
Elayer_regularization_losses
Fmetrics
 regularization_losses
Gnon_trainable_variables
!trainable_variables
ca
VARIABLE_VALUEhidden_layer_3_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEhidden_layer_3_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
?

Hlayers
%	variables
Ilayer_regularization_losses
Jmetrics
&regularization_losses
Knon_trainable_variables
'trainable_variables
a_
VARIABLE_VALUEoutput_layer_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEoutput_layer_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
?

Llayers
+	variables
Mlayer_regularization_losses
Nmetrics
,regularization_losses
Onon_trainable_variables
-trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
#
0
1
2
3
4
 

P0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1
 
 
?

Xlayers
T	variables
Ylayer_regularization_losses
Zmetrics
Uregularization_losses
[non_trainable_variables
Vtrainable_variables
 
 
 

Q0
R1
??
VARIABLE_VALUEAdam/hidden_layer_0_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_0_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_1_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_1_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_2_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_2_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_3_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_3_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/output_layer_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/output_layer_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_0_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_0_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_1_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_1_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_2_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_2_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_3_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_3_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/output_layer_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/output_layer_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
: 
~
serving_default_input_layerPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerhidden_layer_0_1/kernelhidden_layer_0_1/biashidden_layer_1_1/kernelhidden_layer_1_1/biashidden_layer_2_1/kernelhidden_layer_2_1/biashidden_layer_3_1/kernelhidden_layer_3_1/biasoutput_layer_1/kerneloutput_layer_1/bias*
Tin
2*-
f(R&
$__inference_signature_wrapper_451273*
Tout
2*'
_output_shapes
:?????????*-
_gradient_op_typePartitionedCall-451548**
config_proto

CPU

GPU 2J 8
O
saver_filenamePlaceholder*
shape: *
_output_shapes
: *
dtype0
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+hidden_layer_0_1/kernel/Read/ReadVariableOp)hidden_layer_0_1/bias/Read/ReadVariableOp+hidden_layer_1_1/kernel/Read/ReadVariableOp)hidden_layer_1_1/bias/Read/ReadVariableOp+hidden_layer_2_1/kernel/Read/ReadVariableOp)hidden_layer_2_1/bias/Read/ReadVariableOp+hidden_layer_3_1/kernel/Read/ReadVariableOp)hidden_layer_3_1/bias/Read/ReadVariableOp)output_layer_1/kernel/Read/ReadVariableOp'output_layer_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2Adam/hidden_layer_0_1/kernel/m/Read/ReadVariableOp0Adam/hidden_layer_0_1/bias/m/Read/ReadVariableOp2Adam/hidden_layer_1_1/kernel/m/Read/ReadVariableOp0Adam/hidden_layer_1_1/bias/m/Read/ReadVariableOp2Adam/hidden_layer_2_1/kernel/m/Read/ReadVariableOp0Adam/hidden_layer_2_1/bias/m/Read/ReadVariableOp2Adam/hidden_layer_3_1/kernel/m/Read/ReadVariableOp0Adam/hidden_layer_3_1/bias/m/Read/ReadVariableOp0Adam/output_layer_1/kernel/m/Read/ReadVariableOp.Adam/output_layer_1/bias/m/Read/ReadVariableOp2Adam/hidden_layer_0_1/kernel/v/Read/ReadVariableOp0Adam/hidden_layer_0_1/bias/v/Read/ReadVariableOp2Adam/hidden_layer_1_1/kernel/v/Read/ReadVariableOp0Adam/hidden_layer_1_1/bias/v/Read/ReadVariableOp2Adam/hidden_layer_2_1/kernel/v/Read/ReadVariableOp0Adam/hidden_layer_2_1/bias/v/Read/ReadVariableOp2Adam/hidden_layer_3_1/kernel/v/Read/ReadVariableOp0Adam/hidden_layer_3_1/bias/v/Read/ReadVariableOp0Adam/output_layer_1/kernel/v/Read/ReadVariableOp.Adam/output_layer_1/bias/v/Read/ReadVariableOpConst*
Tout
2*2
Tin+
)2'	**
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__traced_save_451606*-
_gradient_op_typePartitionedCall-451607*
_output_shapes
: 
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_layer_0_1/kernelhidden_layer_0_1/biashidden_layer_1_1/kernelhidden_layer_1_1/biashidden_layer_2_1/kernelhidden_layer_2_1/biashidden_layer_3_1/kernelhidden_layer_3_1/biasoutput_layer_1/kerneloutput_layer_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/hidden_layer_0_1/kernel/mAdam/hidden_layer_0_1/bias/mAdam/hidden_layer_1_1/kernel/mAdam/hidden_layer_1_1/bias/mAdam/hidden_layer_2_1/kernel/mAdam/hidden_layer_2_1/bias/mAdam/hidden_layer_3_1/kernel/mAdam/hidden_layer_3_1/bias/mAdam/output_layer_1/kernel/mAdam/output_layer_1/bias/mAdam/hidden_layer_0_1/kernel/vAdam/hidden_layer_0_1/bias/vAdam/hidden_layer_1_1/kernel/vAdam/hidden_layer_1_1/bias/vAdam/hidden_layer_2_1/kernel/vAdam/hidden_layer_2_1/bias/vAdam/hidden_layer_3_1/kernel/vAdam/hidden_layer_3_1/bias/vAdam/output_layer_1/kernel/vAdam/output_layer_1/bias/v*1
Tin*
(2&*+
f&R$
"__inference__traced_restore_451730*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*
Tout
2*-
_gradient_op_typePartitionedCall-451731??
?
?
$__inference_signature_wrapper_451273
input_layer"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layerstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2**
f%R#
!__inference__wrapped_model_451012*'
_output_shapes
:?????????*-
_gradient_op_typePartitionedCall-451260?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :	 :
 :+ '
%
_user_specified_nameinput_layer: 
?	
?
J__inference_hidden_layer_0_layer_call_and_return_conditional_losses_451392

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_451179
input_layer1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_21
-hidden_layer_3_statefulpartitionedcall_args_11
-hidden_layer_3_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?&hidden_layer_2/StatefulPartitionedCall?&hidden_layer_3/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_layer-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-451035*S
fNRL
J__inference_hidden_layer_0_layer_call_and_return_conditional_losses_451029*'
_output_shapes
:?????????*
Tout
2*
Tin
2**
config_proto

CPU

GPU 2J 8?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_451057*-
_gradient_op_typePartitionedCall-451063?
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-451091*'
_output_shapes
:?????????*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_451085*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8?
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0-hidden_layer_3_statefulpartitionedcall_args_1-hidden_layer_3_statefulpartitionedcall_args_2*S
fNRL
J__inference_hidden_layer_3_layer_call_and_return_conditional_losses_451113*-
_gradient_op_typePartitionedCall-451119**
config_proto

CPU

GPU 2J 8*
Tout
2*
Tin
2*'
_output_shapes
:??????????
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-451146*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*
Tin
2*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_451140*
Tout
2?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall: : : : : : : :	 :
 :+ '
%
_user_specified_nameinput_layer: 
?
?
/__inference_hidden_layer_3_layer_call_fn_451453

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*S
fNRL
J__inference_hidden_layer_3_layer_call_and_return_conditional_losses_451113**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*-
_gradient_op_typePartitionedCall-451119*
Tout
2*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
/__inference_hidden_layer_1_layer_call_fn_451417

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*
Tout
2*
Tin
2*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_451057*-
_gradient_op_typePartitionedCall-451063?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?K
?
__inference__traced_save_451606
file_prefix6
2savev2_hidden_layer_0_1_kernel_read_readvariableop4
0savev2_hidden_layer_0_1_bias_read_readvariableop6
2savev2_hidden_layer_1_1_kernel_read_readvariableop4
0savev2_hidden_layer_1_1_bias_read_readvariableop6
2savev2_hidden_layer_2_1_kernel_read_readvariableop4
0savev2_hidden_layer_2_1_bias_read_readvariableop6
2savev2_hidden_layer_3_1_kernel_read_readvariableop4
0savev2_hidden_layer_3_1_bias_read_readvariableop4
0savev2_output_layer_1_kernel_read_readvariableop2
.savev2_output_layer_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop=
9savev2_adam_hidden_layer_0_1_kernel_m_read_readvariableop;
7savev2_adam_hidden_layer_0_1_bias_m_read_readvariableop=
9savev2_adam_hidden_layer_1_1_kernel_m_read_readvariableop;
7savev2_adam_hidden_layer_1_1_bias_m_read_readvariableop=
9savev2_adam_hidden_layer_2_1_kernel_m_read_readvariableop;
7savev2_adam_hidden_layer_2_1_bias_m_read_readvariableop=
9savev2_adam_hidden_layer_3_1_kernel_m_read_readvariableop;
7savev2_adam_hidden_layer_3_1_bias_m_read_readvariableop;
7savev2_adam_output_layer_1_kernel_m_read_readvariableop9
5savev2_adam_output_layer_1_bias_m_read_readvariableop=
9savev2_adam_hidden_layer_0_1_kernel_v_read_readvariableop;
7savev2_adam_hidden_layer_0_1_bias_v_read_readvariableop=
9savev2_adam_hidden_layer_1_1_kernel_v_read_readvariableop;
7savev2_adam_hidden_layer_1_1_bias_v_read_readvariableop=
9savev2_adam_hidden_layer_2_1_kernel_v_read_readvariableop;
7savev2_adam_hidden_layer_2_1_bias_v_read_readvariableop=
9savev2_adam_hidden_layer_3_1_kernel_v_read_readvariableop;
7savev2_adam_hidden_layer_3_1_bias_v_read_readvariableop;
7savev2_adam_output_layer_1_kernel_v_read_readvariableop9
5savev2_adam_output_layer_1_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5325ed59cb1345a0b9f33649e3f5c4d3/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
value	B :*
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:%*
dtype0?
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:%*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_hidden_layer_0_1_kernel_read_readvariableop0savev2_hidden_layer_0_1_bias_read_readvariableop2savev2_hidden_layer_1_1_kernel_read_readvariableop0savev2_hidden_layer_1_1_bias_read_readvariableop2savev2_hidden_layer_2_1_kernel_read_readvariableop0savev2_hidden_layer_2_1_bias_read_readvariableop2savev2_hidden_layer_3_1_kernel_read_readvariableop0savev2_hidden_layer_3_1_bias_read_readvariableop0savev2_output_layer_1_kernel_read_readvariableop.savev2_output_layer_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_adam_hidden_layer_0_1_kernel_m_read_readvariableop7savev2_adam_hidden_layer_0_1_bias_m_read_readvariableop9savev2_adam_hidden_layer_1_1_kernel_m_read_readvariableop7savev2_adam_hidden_layer_1_1_bias_m_read_readvariableop9savev2_adam_hidden_layer_2_1_kernel_m_read_readvariableop7savev2_adam_hidden_layer_2_1_bias_m_read_readvariableop9savev2_adam_hidden_layer_3_1_kernel_m_read_readvariableop7savev2_adam_hidden_layer_3_1_bias_m_read_readvariableop7savev2_adam_output_layer_1_kernel_m_read_readvariableop5savev2_adam_output_layer_1_bias_m_read_readvariableop9savev2_adam_hidden_layer_0_1_kernel_v_read_readvariableop7savev2_adam_hidden_layer_0_1_bias_v_read_readvariableop9savev2_adam_hidden_layer_1_1_kernel_v_read_readvariableop7savev2_adam_hidden_layer_1_1_bias_v_read_readvariableop9savev2_adam_hidden_layer_2_1_kernel_v_read_readvariableop7savev2_adam_hidden_layer_2_1_bias_v_read_readvariableop9savev2_adam_hidden_layer_3_1_kernel_v_read_readvariableop7savev2_adam_hidden_layer_3_1_bias_v_read_readvariableop7savev2_adam_output_layer_1_kernel_v_read_readvariableop5savev2_adam_output_layer_1_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::: : : : : : : ::::::::::::::::::::: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints: : : : : : :  :! :" :# :$ :% :& :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : 
?	
?
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_451057

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ܑ
?
"__inference__traced_restore_451730
file_prefix,
(assignvariableop_hidden_layer_0_1_kernel,
(assignvariableop_1_hidden_layer_0_1_bias.
*assignvariableop_2_hidden_layer_1_1_kernel,
(assignvariableop_3_hidden_layer_1_1_bias.
*assignvariableop_4_hidden_layer_2_1_kernel,
(assignvariableop_5_hidden_layer_2_1_bias.
*assignvariableop_6_hidden_layer_3_1_kernel,
(assignvariableop_7_hidden_layer_3_1_bias,
(assignvariableop_8_output_layer_1_kernel*
&assignvariableop_9_output_layer_1_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count6
2assignvariableop_17_adam_hidden_layer_0_1_kernel_m4
0assignvariableop_18_adam_hidden_layer_0_1_bias_m6
2assignvariableop_19_adam_hidden_layer_1_1_kernel_m4
0assignvariableop_20_adam_hidden_layer_1_1_bias_m6
2assignvariableop_21_adam_hidden_layer_2_1_kernel_m4
0assignvariableop_22_adam_hidden_layer_2_1_bias_m6
2assignvariableop_23_adam_hidden_layer_3_1_kernel_m4
0assignvariableop_24_adam_hidden_layer_3_1_bias_m4
0assignvariableop_25_adam_output_layer_1_kernel_m2
.assignvariableop_26_adam_output_layer_1_bias_m6
2assignvariableop_27_adam_hidden_layer_0_1_kernel_v4
0assignvariableop_28_adam_hidden_layer_0_1_bias_v6
2assignvariableop_29_adam_hidden_layer_1_1_kernel_v4
0assignvariableop_30_adam_hidden_layer_1_1_bias_v6
2assignvariableop_31_adam_hidden_layer_2_1_kernel_v4
0assignvariableop_32_adam_hidden_layer_2_1_bias_v6
2assignvariableop_33_adam_hidden_layer_3_1_kernel_v4
0assignvariableop_34_adam_hidden_layer_3_1_bias_v4
0assignvariableop_35_adam_output_layer_1_kernel_v2
.assignvariableop_36_adam_output_layer_1_bias_v
identity_38??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*?
value?B?%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:%?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:%?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp(assignvariableop_hidden_layer_0_1_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0?
AssignVariableOp_1AssignVariableOp(assignvariableop_1_hidden_layer_0_1_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp*assignvariableop_2_hidden_layer_1_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0?
AssignVariableOp_3AssignVariableOp(assignvariableop_3_hidden_layer_1_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_hidden_layer_2_1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0?
AssignVariableOp_5AssignVariableOp(assignvariableop_5_hidden_layer_2_1_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp*assignvariableop_6_hidden_layer_3_1_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0?
AssignVariableOp_7AssignVariableOp(assignvariableop_7_hidden_layer_3_1_biasIdentity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_output_layer_1_kernelIdentity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0?
AssignVariableOp_9AssignVariableOp&assignvariableop_9_output_layer_1_biasIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0*
_output_shapes
 *
dtype0	P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0*
_output_shapes
 *
dtype0P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0{
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:{
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0?
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_hidden_layer_0_1_kernel_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_hidden_layer_0_1_bias_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0?
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_hidden_layer_1_1_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_hidden_layer_1_1_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype0P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0?
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_hidden_layer_2_1_kernel_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp0assignvariableop_22_adam_hidden_layer_2_1_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype0P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_hidden_layer_3_1_kernel_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp0assignvariableop_24_adam_hidden_layer_3_1_bias_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_output_layer_1_kernel_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_adam_output_layer_1_bias_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0?
AssignVariableOp_27AssignVariableOp2assignvariableop_27_adam_hidden_layer_0_1_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype0P
Identity_28IdentityRestoreV2:tensors:28*
_output_shapes
:*
T0?
AssignVariableOp_28AssignVariableOp0assignvariableop_28_adam_hidden_layer_0_1_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype0P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0?
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_hidden_layer_1_1_kernel_vIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp0assignvariableop_30_adam_hidden_layer_1_1_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype0P
Identity_31IdentityRestoreV2:tensors:31*
_output_shapes
:*
T0?
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_hidden_layer_2_1_kernel_vIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
_output_shapes
:*
T0?
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_hidden_layer_2_1_bias_vIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
_output_shapes
:*
T0?
AssignVariableOp_33AssignVariableOp2assignvariableop_33_adam_hidden_layer_3_1_kernel_vIdentity_33:output:0*
_output_shapes
 *
dtype0P
Identity_34IdentityRestoreV2:tensors:34*
_output_shapes
:*
T0?
AssignVariableOp_34AssignVariableOp0assignvariableop_34_adam_hidden_layer_3_1_bias_vIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp0assignvariableop_35_adam_output_layer_1_kernel_vIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
_output_shapes
:*
T0?
AssignVariableOp_36AssignVariableOp.assignvariableop_36_adam_output_layer_1_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype0?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_38Identity_38:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_451238

inputs1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_21
-hidden_layer_3_statefulpartitionedcall_args_11
-hidden_layer_3_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?&hidden_layer_2/StatefulPartitionedCall?&hidden_layer_3/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputs-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*-
_gradient_op_typePartitionedCall-451035**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*S
fNRL
J__inference_hidden_layer_0_layer_call_and_return_conditional_losses_451029?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*
Tout
2*-
_gradient_op_typePartitionedCall-451063**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_451057*'
_output_shapes
:?????????*
Tin
2?
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2*-
_gradient_op_typePartitionedCall-451091*'
_output_shapes
:?????????*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_451085?
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0-hidden_layer_3_statefulpartitionedcall_args_1-hidden_layer_3_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-451119*'
_output_shapes
:?????????*S
fNRL
J__inference_hidden_layer_3_layer_call_and_return_conditional_losses_451113*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*
Tin
2**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_451140*-
_gradient_op_typePartitionedCall-451146*'
_output_shapes
:?????????*
Tout
2?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall: :	 :
 :& "
 
_user_specified_nameinputs: : : : : : : 
?	
?
J__inference_hidden_layer_3_layer_call_and_return_conditional_losses_451113

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_451428

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?	
?
H__inference_output_layer_layer_call_and_return_conditional_losses_451463

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
-__inference_sequential_1_layer_call_fn_451366

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tout
2*-
_gradient_op_typePartitionedCall-451202**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_451201*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :	 :
 :& "
 
_user_specified_nameinputs: : : : : 
?	
?
J__inference_hidden_layer_0_layer_call_and_return_conditional_losses_451029

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?2
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_451313

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource1
-hidden_layer_3_matmul_readvariableop_resource2
.hidden_layer_3_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity??%hidden_layer_0/BiasAdd/ReadVariableOp?$hidden_layer_0/MatMul/ReadVariableOp?%hidden_layer_1/BiasAdd/ReadVariableOp?$hidden_layer_1/MatMul/ReadVariableOp?%hidden_layer_2/BiasAdd/ReadVariableOp?$hidden_layer_2/MatMul/ReadVariableOp?%hidden_layer_3/BiasAdd/ReadVariableOp?$hidden_layer_3/MatMul/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOp?
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
hidden_layer_0/MatMulMatMulinputs,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
hidden_layer_1/MatMulMatMul!hidden_layer_0/Relu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0n
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
hidden_layer_2/MatMulMatMul!hidden_layer_1/Relu:activations:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0n
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
$hidden_layer_3/MatMul/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
hidden_layer_3/MatMulMatMul!hidden_layer_2/Relu:activations:0,hidden_layer_3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
%hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
hidden_layer_3/BiasAddBiasAddhidden_layer_3/MatMul:product:0-hidden_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
hidden_layer_3/ReluReluhidden_layer_3/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
output_layer/MatMulMatMul!hidden_layer_3/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentityoutput_layer/BiasAdd:output:0&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp&^hidden_layer_3/BiasAdd/ReadVariableOp%^hidden_layer_3/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2N
%hidden_layer_3/BiasAdd/ReadVariableOp%hidden_layer_3/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp2L
$hidden_layer_3/MatMul/ReadVariableOp$hidden_layer_3/MatMul/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_451201

inputs1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_21
-hidden_layer_3_statefulpartitionedcall_args_11
-hidden_layer_3_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?&hidden_layer_2/StatefulPartitionedCall?&hidden_layer_3/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinputs-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*S
fNRL
J__inference_hidden_layer_0_layer_call_and_return_conditional_losses_451029*
Tout
2*
Tin
2**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-451035*'
_output_shapes
:??????????
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*
Tin
2*-
_gradient_op_typePartitionedCall-451063**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*
Tout
2*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_451057?
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-451091*'
_output_shapes
:?????????*
Tout
2*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_451085**
config_proto

CPU

GPU 2J 8*
Tin
2?
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0-hidden_layer_3_statefulpartitionedcall_args_1-hidden_layer_3_statefulpartitionedcall_args_2*
Tin
2*S
fNRL
J__inference_hidden_layer_3_layer_call_and_return_conditional_losses_451113**
config_proto

CPU

GPU 2J 8*
Tout
2*-
_gradient_op_typePartitionedCall-451119*'
_output_shapes
:??????????
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_451140*
Tout
2*-
_gradient_op_typePartitionedCall-451146*
Tin
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:??????????
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall: : : :	 :
 :& "
 
_user_specified_nameinputs: : : : : 
?
?
-__inference_sequential_1_layer_call_fn_451252
input_layer"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layerstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_451238*'
_output_shapes
:?????????*-
_gradient_op_typePartitionedCall-451239**
config_proto

CPU

GPU 2J 8*
Tout
2*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_layer: : : : : : : : :	 :
 
?	
?
J__inference_hidden_layer_3_layer_call_and_return_conditional_losses_451446

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
/__inference_hidden_layer_0_layer_call_fn_451399

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-451035*'
_output_shapes
:?????????*
Tout
2*S
fNRL
J__inference_hidden_layer_0_layer_call_and_return_conditional_losses_451029*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
-__inference_output_layer_layer_call_fn_451470

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-451146*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_451140*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
-__inference_sequential_1_layer_call_fn_451381

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*-
_gradient_op_typePartitionedCall-451239**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_451238*'
_output_shapes
:?????????*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 
?	
?
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_451410

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?=
?	
!__inference__wrapped_model_451012
input_layer>
:sequential_1_hidden_layer_0_matmul_readvariableop_resource?
;sequential_1_hidden_layer_0_biasadd_readvariableop_resource>
:sequential_1_hidden_layer_1_matmul_readvariableop_resource?
;sequential_1_hidden_layer_1_biasadd_readvariableop_resource>
:sequential_1_hidden_layer_2_matmul_readvariableop_resource?
;sequential_1_hidden_layer_2_biasadd_readvariableop_resource>
:sequential_1_hidden_layer_3_matmul_readvariableop_resource?
;sequential_1_hidden_layer_3_biasadd_readvariableop_resource<
8sequential_1_output_layer_matmul_readvariableop_resource=
9sequential_1_output_layer_biasadd_readvariableop_resource
identity??2sequential_1/hidden_layer_0/BiasAdd/ReadVariableOp?1sequential_1/hidden_layer_0/MatMul/ReadVariableOp?2sequential_1/hidden_layer_1/BiasAdd/ReadVariableOp?1sequential_1/hidden_layer_1/MatMul/ReadVariableOp?2sequential_1/hidden_layer_2/BiasAdd/ReadVariableOp?1sequential_1/hidden_layer_2/MatMul/ReadVariableOp?2sequential_1/hidden_layer_3/BiasAdd/ReadVariableOp?1sequential_1/hidden_layer_3/MatMul/ReadVariableOp?0sequential_1/output_layer/BiasAdd/ReadVariableOp?/sequential_1/output_layer/MatMul/ReadVariableOp?
1sequential_1/hidden_layer_0/MatMul/ReadVariableOpReadVariableOp:sequential_1_hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
"sequential_1/hidden_layer_0/MatMulMatMulinput_layer9sequential_1/hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
2sequential_1/hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp;sequential_1_hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
#sequential_1/hidden_layer_0/BiasAddBiasAdd,sequential_1/hidden_layer_0/MatMul:product:0:sequential_1/hidden_layer_0/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
 sequential_1/hidden_layer_0/ReluRelu,sequential_1/hidden_layer_0/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
1sequential_1/hidden_layer_1/MatMul/ReadVariableOpReadVariableOp:sequential_1_hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
"sequential_1/hidden_layer_1/MatMulMatMul.sequential_1/hidden_layer_0/Relu:activations:09sequential_1/hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
2sequential_1/hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp;sequential_1_hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
#sequential_1/hidden_layer_1/BiasAddBiasAdd,sequential_1/hidden_layer_1/MatMul:product:0:sequential_1/hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 sequential_1/hidden_layer_1/ReluRelu,sequential_1/hidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
1sequential_1/hidden_layer_2/MatMul/ReadVariableOpReadVariableOp:sequential_1_hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
"sequential_1/hidden_layer_2/MatMulMatMul.sequential_1/hidden_layer_1/Relu:activations:09sequential_1/hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
2sequential_1/hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp;sequential_1_hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
#sequential_1/hidden_layer_2/BiasAddBiasAdd,sequential_1/hidden_layer_2/MatMul:product:0:sequential_1/hidden_layer_2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
 sequential_1/hidden_layer_2/ReluRelu,sequential_1/hidden_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
1sequential_1/hidden_layer_3/MatMul/ReadVariableOpReadVariableOp:sequential_1_hidden_layer_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
"sequential_1/hidden_layer_3/MatMulMatMul.sequential_1/hidden_layer_2/Relu:activations:09sequential_1/hidden_layer_3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
2sequential_1/hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp;sequential_1_hidden_layer_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
#sequential_1/hidden_layer_3/BiasAddBiasAdd,sequential_1/hidden_layer_3/MatMul:product:0:sequential_1/hidden_layer_3/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
 sequential_1/hidden_layer_3/ReluRelu,sequential_1/hidden_layer_3/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
/sequential_1/output_layer/MatMul/ReadVariableOpReadVariableOp8sequential_1_output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
 sequential_1/output_layer/MatMulMatMul.sequential_1/hidden_layer_3/Relu:activations:07sequential_1/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
0sequential_1/output_layer/BiasAdd/ReadVariableOpReadVariableOp9sequential_1_output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
!sequential_1/output_layer/BiasAddBiasAdd*sequential_1/output_layer/MatMul:product:08sequential_1/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentity*sequential_1/output_layer/BiasAdd:output:03^sequential_1/hidden_layer_0/BiasAdd/ReadVariableOp2^sequential_1/hidden_layer_0/MatMul/ReadVariableOp3^sequential_1/hidden_layer_1/BiasAdd/ReadVariableOp2^sequential_1/hidden_layer_1/MatMul/ReadVariableOp3^sequential_1/hidden_layer_2/BiasAdd/ReadVariableOp2^sequential_1/hidden_layer_2/MatMul/ReadVariableOp3^sequential_1/hidden_layer_3/BiasAdd/ReadVariableOp2^sequential_1/hidden_layer_3/MatMul/ReadVariableOp1^sequential_1/output_layer/BiasAdd/ReadVariableOp0^sequential_1/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2d
0sequential_1/output_layer/BiasAdd/ReadVariableOp0sequential_1/output_layer/BiasAdd/ReadVariableOp2b
/sequential_1/output_layer/MatMul/ReadVariableOp/sequential_1/output_layer/MatMul/ReadVariableOp2h
2sequential_1/hidden_layer_3/BiasAdd/ReadVariableOp2sequential_1/hidden_layer_3/BiasAdd/ReadVariableOp2f
1sequential_1/hidden_layer_0/MatMul/ReadVariableOp1sequential_1/hidden_layer_0/MatMul/ReadVariableOp2f
1sequential_1/hidden_layer_2/MatMul/ReadVariableOp1sequential_1/hidden_layer_2/MatMul/ReadVariableOp2h
2sequential_1/hidden_layer_2/BiasAdd/ReadVariableOp2sequential_1/hidden_layer_2/BiasAdd/ReadVariableOp2h
2sequential_1/hidden_layer_1/BiasAdd/ReadVariableOp2sequential_1/hidden_layer_1/BiasAdd/ReadVariableOp2h
2sequential_1/hidden_layer_0/BiasAdd/ReadVariableOp2sequential_1/hidden_layer_0/BiasAdd/ReadVariableOp2f
1sequential_1/hidden_layer_1/MatMul/ReadVariableOp1sequential_1/hidden_layer_1/MatMul/ReadVariableOp2f
1sequential_1/hidden_layer_3/MatMul/ReadVariableOp1sequential_1/hidden_layer_3/MatMul/ReadVariableOp:
 :+ '
%
_user_specified_nameinput_layer: : : : : : : : :	 
?	
?
H__inference_output_layer_layer_call_and_return_conditional_losses_451140

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_451085

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0P
ReluReluBiasAdd:output:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
-__inference_sequential_1_layer_call_fn_451215
input_layer"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layerstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tout
2*
Tin
2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_451201*-
_gradient_op_typePartitionedCall-451202?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:
 :+ '
%
_user_specified_nameinput_layer: : : : : : : : :	 
?
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_451158
input_layer1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_21
-hidden_layer_3_statefulpartitionedcall_args_11
-hidden_layer_3_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?&hidden_layer_2/StatefulPartitionedCall?&hidden_layer_3/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCallinput_layer-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-451035*S
fNRL
J__inference_hidden_layer_0_layer_call_and_return_conditional_losses_451029*
Tin
2*
Tout
2?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tout
2*-
_gradient_op_typePartitionedCall-451063*
Tin
2*S
fNRL
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_451057*'
_output_shapes
:??????????
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*-
_gradient_op_typePartitionedCall-451091*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_451085*'
_output_shapes
:?????????*
Tout
2*
Tin
2?
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0-hidden_layer_3_statefulpartitionedcall_args_1-hidden_layer_3_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????*S
fNRL
J__inference_hidden_layer_3_layer_call_and_return_conditional_losses_451113*-
_gradient_op_typePartitionedCall-451119*
Tin
2?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-451146*
Tout
2*
Tin
2*Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_451140*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????*
T0"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:
 :+ '
%
_user_specified_nameinput_layer: : : : : : : : :	 
?2
?
H__inference_sequential_1_layer_call_and_return_conditional_losses_451351

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource1
-hidden_layer_3_matmul_readvariableop_resource2
.hidden_layer_3_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity??%hidden_layer_0/BiasAdd/ReadVariableOp?$hidden_layer_0/MatMul/ReadVariableOp?%hidden_layer_1/BiasAdd/ReadVariableOp?$hidden_layer_1/MatMul/ReadVariableOp?%hidden_layer_2/BiasAdd/ReadVariableOp?$hidden_layer_2/MatMul/ReadVariableOp?%hidden_layer_3/BiasAdd/ReadVariableOp?$hidden_layer_3/MatMul/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOp?
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
hidden_layer_0/MatMulMatMulinputs,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0n
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
hidden_layer_1/MatMulMatMul!hidden_layer_0/Relu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
hidden_layer_2/MatMulMatMul!hidden_layer_1/Relu:activations:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0n
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*'
_output_shapes
:?????????*
T0?
$hidden_layer_3/MatMul/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0?
hidden_layer_3/MatMulMatMul!hidden_layer_2/Relu:activations:0,hidden_layer_3/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
%hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
hidden_layer_3/BiasAddBiasAddhidden_layer_3/MatMul:product:0-hidden_layer_3/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0n
hidden_layer_3/ReluReluhidden_layer_3/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:?
output_layer/MatMulMatMul!hidden_layer_3/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????*
T0?
IdentityIdentityoutput_layer/BiasAdd:output:0&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp&^hidden_layer_3/BiasAdd/ReadVariableOp%^hidden_layer_3/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2N
%hidden_layer_3/BiasAdd/ReadVariableOp%hidden_layer_3/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp2L
$hidden_layer_3/MatMul/ReadVariableOp$hidden_layer_3/MatMul/ReadVariableOp:
 :& "
 
_user_specified_nameinputs: : : : : : : : :	 
?
?
/__inference_hidden_layer_2_layer_call_fn_451435

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*'
_output_shapes
:?????????*S
fNRL
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_451085*-
_gradient_op_typePartitionedCall-451091**
config_proto

CPU

GPU 2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_layer4
serving_default_input_layer:0?????????@
output_layer0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?,
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
*p&call_and_return_all_conditional_losses
q_default_save_signature
r__call__"?)
_tf_keras_sequential?({"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 6]}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_3", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 6]}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_3", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": ["mean_squared_error"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*s&call_and_return_all_conditional_losses
t__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "input_layer", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 6], "config": {"batch_input_shape": [null, 6], "dtype": "float32", "sparse": false, "name": "input_layer"}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*u&call_and_return_all_conditional_losses
v__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}}
?

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
*y&call_and_return_all_conditional_losses
z__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}}
?

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
*{&call_and_return_all_conditional_losses
|__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_3", "trainable": true, "dtype": "float32", "units": 6, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}}
?

)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
*}&call_and_return_all_conditional_losses
~__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}}
?
/iter

0beta_1

1beta_2
	2decay
3learning_ratem\m]m^m_m`ma#mb$mc)md*mevfvgvhvivjvk#vl$vm)vn*vo"
	optimizer
f
0
1
2
3
4
5
#6
$7
)8
*9"
trackable_list_wrapper
f
0
1
2
3
4
5
#6
$7
)8
*9"
trackable_list_wrapper
 "
trackable_list_wrapper
?

4layers
	variables
5layer_regularization_losses
6metrics
7non_trainable_variables
	trainable_variables

regularization_losses
r__call__
q_default_save_signature
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

8layers
	variables
9layer_regularization_losses
:metrics
regularization_losses
;non_trainable_variables
trainable_variables
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
):'2hidden_layer_0_1/kernel
#:!2hidden_layer_0_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

<layers
	variables
=layer_regularization_losses
>metrics
regularization_losses
?non_trainable_variables
trainable_variables
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
):'2hidden_layer_1_1/kernel
#:!2hidden_layer_1_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

@layers
	variables
Alayer_regularization_losses
Bmetrics
regularization_losses
Cnon_trainable_variables
trainable_variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
):'2hidden_layer_2_1/kernel
#:!2hidden_layer_2_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

Dlayers
	variables
Elayer_regularization_losses
Fmetrics
 regularization_losses
Gnon_trainable_variables
!trainable_variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
):'2hidden_layer_3_1/kernel
#:!2hidden_layer_3_1/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?

Hlayers
%	variables
Ilayer_regularization_losses
Jmetrics
&regularization_losses
Knon_trainable_variables
'trainable_variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
':%2output_layer_1/kernel
!:2output_layer_1/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?

Llayers
+	variables
Mlayer_regularization_losses
Nmetrics
,regularization_losses
Onon_trainable_variables
-trainable_variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Qtotal
	Rcount
S
_fn_kwargs
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "mean_squared_error", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mean_squared_error", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Xlayers
T	variables
Ylayer_regularization_losses
Zmetrics
Uregularization_losses
[non_trainable_variables
Vtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.:,2Adam/hidden_layer_0_1/kernel/m
(:&2Adam/hidden_layer_0_1/bias/m
.:,2Adam/hidden_layer_1_1/kernel/m
(:&2Adam/hidden_layer_1_1/bias/m
.:,2Adam/hidden_layer_2_1/kernel/m
(:&2Adam/hidden_layer_2_1/bias/m
.:,2Adam/hidden_layer_3_1/kernel/m
(:&2Adam/hidden_layer_3_1/bias/m
,:*2Adam/output_layer_1/kernel/m
&:$2Adam/output_layer_1/bias/m
.:,2Adam/hidden_layer_0_1/kernel/v
(:&2Adam/hidden_layer_0_1/bias/v
.:,2Adam/hidden_layer_1_1/kernel/v
(:&2Adam/hidden_layer_1_1/bias/v
.:,2Adam/hidden_layer_2_1/kernel/v
(:&2Adam/hidden_layer_2_1/bias/v
.:,2Adam/hidden_layer_3_1/kernel/v
(:&2Adam/hidden_layer_3_1/bias/v
,:*2Adam/output_layer_1/kernel/v
&:$2Adam/output_layer_1/bias/v
?2?
H__inference_sequential_1_layer_call_and_return_conditional_losses_451313
H__inference_sequential_1_layer_call_and_return_conditional_losses_451351
H__inference_sequential_1_layer_call_and_return_conditional_losses_451179
H__inference_sequential_1_layer_call_and_return_conditional_losses_451158?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_451012?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
input_layer?????????
?2?
-__inference_sequential_1_layer_call_fn_451215
-__inference_sequential_1_layer_call_fn_451381
-__inference_sequential_1_layer_call_fn_451366
-__inference_sequential_1_layer_call_fn_451252?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
J__inference_hidden_layer_0_layer_call_and_return_conditional_losses_451392?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_hidden_layer_0_layer_call_fn_451399?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_451410?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_hidden_layer_1_layer_call_fn_451417?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_451428?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_hidden_layer_2_layer_call_fn_451435?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_hidden_layer_3_layer_call_and_return_conditional_losses_451446?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_hidden_layer_3_layer_call_fn_451453?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_output_layer_layer_call_and_return_conditional_losses_451463?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_output_layer_layer_call_fn_451470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
7B5
$__inference_signature_wrapper_451273input_layer
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
-__inference_sequential_1_layer_call_fn_451215d
#$)*<?9
2?/
%?"
input_layer?????????
p

 
? "???????????
-__inference_sequential_1_layer_call_fn_451366_
#$)*7?4
-?*
 ?
inputs?????????
p

 
? "???????????
-__inference_sequential_1_layer_call_fn_451381_
#$)*7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
/__inference_hidden_layer_2_layer_call_fn_451435O/?,
%?"
 ?
inputs?????????
? "???????????
J__inference_hidden_layer_3_layer_call_and_return_conditional_losses_451446\#$/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
H__inference_output_layer_layer_call_and_return_conditional_losses_451463\)*/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_451351l
#$)*7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_1_layer_call_and_return_conditional_losses_451313l
#$)*7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
-__inference_output_layer_layer_call_fn_451470O)*/?,
%?"
 ?
inputs?????????
? "???????????
/__inference_hidden_layer_3_layer_call_fn_451453O#$/?,
%?"
 ?
inputs?????????
? "???????????
J__inference_hidden_layer_0_layer_call_and_return_conditional_losses_451392\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
/__inference_hidden_layer_1_layer_call_fn_451417O/?,
%?"
 ?
inputs?????????
? "???????????
H__inference_sequential_1_layer_call_and_return_conditional_losses_451179q
#$)*<?9
2?/
%?"
input_layer?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_hidden_layer_1_layer_call_and_return_conditional_losses_451410\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
!__inference__wrapped_model_451012
#$)*4?1
*?'
%?"
input_layer?????????
? ";?8
6
output_layer&?#
output_layer??????????
J__inference_hidden_layer_2_layer_call_and_return_conditional_losses_451428\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
/__inference_hidden_layer_0_layer_call_fn_451399O/?,
%?"
 ?
inputs?????????
? "???????????
H__inference_sequential_1_layer_call_and_return_conditional_losses_451158q
#$)*<?9
2?/
%?"
input_layer?????????
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_1_layer_call_fn_451252d
#$)*<?9
2?/
%?"
input_layer?????????
p 

 
? "???????????
$__inference_signature_wrapper_451273?
#$)*C?@
? 
9?6
4
input_layer%?"
input_layer?????????";?8
6
output_layer&?#
output_layer?????????