??
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
shapeshape?"serve*2.0.02unknown8??
?
hidden_layer_0_79/kernelVarHandleOp*
dtype0*)
shared_namehidden_layer_0_79/kernel*
shape:
??*
_output_shapes
: 
?
,hidden_layer_0_79/kernel/Read/ReadVariableOpReadVariableOphidden_layer_0_79/kernel* 
_output_shapes
:
??*
dtype0
?
hidden_layer_0_79/biasVarHandleOp*
shape:?*
dtype0*
_output_shapes
: *'
shared_namehidden_layer_0_79/bias
~
*hidden_layer_0_79/bias/Read/ReadVariableOpReadVariableOphidden_layer_0_79/bias*
_output_shapes	
:?*
dtype0
?
hidden_layer_1_78/kernelVarHandleOp*
shape:
??*
dtype0*)
shared_namehidden_layer_1_78/kernel*
_output_shapes
: 
?
,hidden_layer_1_78/kernel/Read/ReadVariableOpReadVariableOphidden_layer_1_78/kernel* 
_output_shapes
:
??*
dtype0
?
hidden_layer_1_78/biasVarHandleOp*
shape:?*
_output_shapes
: *
dtype0*'
shared_namehidden_layer_1_78/bias
~
*hidden_layer_1_78/bias/Read/ReadVariableOpReadVariableOphidden_layer_1_78/bias*
dtype0*
_output_shapes	
:?
?
output_layer_79/kernelVarHandleOp*
dtype0*'
shared_nameoutput_layer_79/kernel*
_output_shapes
: *
shape:	?

?
*output_layer_79/kernel/Read/ReadVariableOpReadVariableOpoutput_layer_79/kernel*
_output_shapes
:	?
*
dtype0
?
output_layer_79/biasVarHandleOp*
_output_shapes
: *
dtype0*%
shared_nameoutput_layer_79/bias*
shape:

y
(output_layer_79/bias/Read/ReadVariableOpReadVariableOpoutput_layer_79/bias*
dtype0*
_output_shapes
:

f
	Adam/iterVarHandleOp*
shape: *
_output_shapes
: *
dtype0	*
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
shared_nameAdam/beta_1*
shape: *
dtype0*
_output_shapes
: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
dtype0*
shared_nameAdam/beta_2*
_output_shapes
: *
shape: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
shared_name
Adam/decay*
shape: *
dtype0
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*#
shared_nameAdam/learning_rate*
_output_shapes
: *
shape: *
dtype0
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shared_nametotal*
shape: *
_output_shapes
: *
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
shape: *
shared_namecount*
_output_shapes
: *
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/hidden_layer_0_79/kernel/mVarHandleOp*
shape:
??*
_output_shapes
: *0
shared_name!Adam/hidden_layer_0_79/kernel/m*
dtype0
?
3Adam/hidden_layer_0_79/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_0_79/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/hidden_layer_0_79/bias/mVarHandleOp*
dtype0*
shape:?*
_output_shapes
: *.
shared_nameAdam/hidden_layer_0_79/bias/m
?
1Adam/hidden_layer_0_79/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_0_79/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/hidden_layer_1_78/kernel/mVarHandleOp*
shape:
??*
dtype0*
_output_shapes
: *0
shared_name!Adam/hidden_layer_1_78/kernel/m
?
3Adam/hidden_layer_1_78/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_1_78/kernel/m*
dtype0* 
_output_shapes
:
??
?
Adam/hidden_layer_1_78/bias/mVarHandleOp*
shape:?*
dtype0*
_output_shapes
: *.
shared_nameAdam/hidden_layer_1_78/bias/m
?
1Adam/hidden_layer_1_78/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_1_78/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/output_layer_79/kernel/mVarHandleOp*.
shared_nameAdam/output_layer_79/kernel/m*
dtype0*
_output_shapes
: *
shape:	?

?
1Adam/output_layer_79/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output_layer_79/kernel/m*
dtype0*
_output_shapes
:	?

?
Adam/output_layer_79/bias/mVarHandleOp*
shape:
*,
shared_nameAdam/output_layer_79/bias/m*
_output_shapes
: *
dtype0
?
/Adam/output_layer_79/bias/m/Read/ReadVariableOpReadVariableOpAdam/output_layer_79/bias/m*
dtype0*
_output_shapes
:

?
Adam/hidden_layer_0_79/kernel/vVarHandleOp*0
shared_name!Adam/hidden_layer_0_79/kernel/v*
dtype0*
shape:
??*
_output_shapes
: 
?
3Adam/hidden_layer_0_79/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_0_79/kernel/v*
dtype0* 
_output_shapes
:
??
?
Adam/hidden_layer_0_79/bias/vVarHandleOp*
dtype0*
shape:?*.
shared_nameAdam/hidden_layer_0_79/bias/v*
_output_shapes
: 
?
1Adam/hidden_layer_0_79/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_0_79/bias/v*
dtype0*
_output_shapes	
:?
?
Adam/hidden_layer_1_78/kernel/vVarHandleOp*0
shared_name!Adam/hidden_layer_1_78/kernel/v*
shape:
??*
_output_shapes
: *
dtype0
?
3Adam/hidden_layer_1_78/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_1_78/kernel/v*
dtype0* 
_output_shapes
:
??
?
Adam/hidden_layer_1_78/bias/vVarHandleOp*.
shared_nameAdam/hidden_layer_1_78/bias/v*
shape:?*
dtype0*
_output_shapes
: 
?
1Adam/hidden_layer_1_78/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_layer_1_78/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/output_layer_79/kernel/vVarHandleOp*.
shared_nameAdam/output_layer_79/kernel/v*
shape:	?
*
dtype0*
_output_shapes
: 
?
1Adam/output_layer_79/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output_layer_79/kernel/v*
dtype0*
_output_shapes
:	?

?
Adam/output_layer_79/bias/vVarHandleOp*,
shared_nameAdam/output_layer_79/bias/v*
shape:
*
dtype0*
_output_shapes
: 
?
/Adam/output_layer_79/bias/v/Read/ReadVariableOpReadVariableOpAdam/output_layer_79/bias/v*
dtype0*
_output_shapes
:


NoOpNoOp
?+
ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *?+
value?+B?+ B?+
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
R
!	variables
"regularization_losses
#trainable_variables
$	keras_api
h

%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
?
+iter

,beta_1

-beta_2
	.decay
/learning_ratemXmYmZm[%m\&m]v^v_v`va%vb&vc
*
0
1
2
3
%4
&5
 
*
0
1
2
3
%4
&5
?
	variables
	regularization_losses

trainable_variables

0layers
1metrics
2non_trainable_variables
3layer_regularization_losses
 
 
 
 
?
	variables
regularization_losses
trainable_variables

4layers
5metrics
6non_trainable_variables
7layer_regularization_losses
 
 
 
?
	variables
regularization_losses
trainable_variables

8layers
9metrics
:non_trainable_variables
;layer_regularization_losses
db
VARIABLE_VALUEhidden_layer_0_79/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEhidden_layer_0_79/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
regularization_losses
trainable_variables

<layers
=metrics
>non_trainable_variables
?layer_regularization_losses
db
VARIABLE_VALUEhidden_layer_1_78/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEhidden_layer_1_78/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
regularization_losses
trainable_variables

@layers
Ametrics
Bnon_trainable_variables
Clayer_regularization_losses
 
 
 
?
!	variables
"regularization_losses
#trainable_variables

Dlayers
Emetrics
Fnon_trainable_variables
Glayer_regularization_losses
b`
VARIABLE_VALUEoutput_layer_79/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEoutput_layer_79/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
?
'	variables
(regularization_losses
)trainable_variables

Hlayers
Imetrics
Jnon_trainable_variables
Klayer_regularization_losses
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

L0
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
 
x
	Mtotal
	Ncount
O
_fn_kwargs
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1
 
 
?
P	variables
Qregularization_losses
Rtrainable_variables

Tlayers
Umetrics
Vnon_trainable_variables
Wlayer_regularization_losses
 
 

M0
N1
 
??
VARIABLE_VALUEAdam/hidden_layer_0_79/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_0_79/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_1_78/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_1_78/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/output_layer_79/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/output_layer_79/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_0_79/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_0_79/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_1_78/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/hidden_layer_1_78/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/output_layer_79/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/output_layer_79/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
!serving_default_input_layer_inputPlaceholder*+
_output_shapes
:?????????* 
shape:?????????*
dtype0
?
StatefulPartitionedCallStatefulPartitionedCall!serving_default_input_layer_inputhidden_layer_0_79/kernelhidden_layer_0_79/biashidden_layer_1_78/kernelhidden_layer_1_78/biasoutput_layer_79/kerneloutput_layer_79/bias*
Tout
2*'
_output_shapes
:?????????
*
Tin
	2**
config_proto

GPU 

CPU2J 8*.
f)R'
%__inference_signature_wrapper_4792959*.
_gradient_op_typePartitionedCall-4793208
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,hidden_layer_0_79/kernel/Read/ReadVariableOp*hidden_layer_0_79/bias/Read/ReadVariableOp,hidden_layer_1_78/kernel/Read/ReadVariableOp*hidden_layer_1_78/bias/Read/ReadVariableOp*output_layer_79/kernel/Read/ReadVariableOp(output_layer_79/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp3Adam/hidden_layer_0_79/kernel/m/Read/ReadVariableOp1Adam/hidden_layer_0_79/bias/m/Read/ReadVariableOp3Adam/hidden_layer_1_78/kernel/m/Read/ReadVariableOp1Adam/hidden_layer_1_78/bias/m/Read/ReadVariableOp1Adam/output_layer_79/kernel/m/Read/ReadVariableOp/Adam/output_layer_79/bias/m/Read/ReadVariableOp3Adam/hidden_layer_0_79/kernel/v/Read/ReadVariableOp1Adam/hidden_layer_0_79/bias/v/Read/ReadVariableOp3Adam/hidden_layer_1_78/kernel/v/Read/ReadVariableOp1Adam/hidden_layer_1_78/bias/v/Read/ReadVariableOp1Adam/output_layer_79/kernel/v/Read/ReadVariableOp/Adam/output_layer_79/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2**
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_save_4793254*.
_gradient_op_typePartitionedCall-4793255*
_output_shapes
: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_layer_0_79/kernelhidden_layer_0_79/biashidden_layer_1_78/kernelhidden_layer_1_78/biasoutput_layer_79/kerneloutput_layer_79/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/hidden_layer_0_79/kernel/mAdam/hidden_layer_0_79/bias/mAdam/hidden_layer_1_78/kernel/mAdam/hidden_layer_1_78/bias/mAdam/output_layer_79/kernel/mAdam/output_layer_79/bias/mAdam/hidden_layer_0_79/kernel/vAdam/hidden_layer_0_79/bias/vAdam/hidden_layer_1_78/kernel/vAdam/hidden_layer_1_78/bias/vAdam/output_layer_79/kernel/vAdam/output_layer_79/bias/v*%
Tin
2*.
_gradient_op_typePartitionedCall-4793343*,
f'R%
#__inference__traced_restore_4793342**
config_proto

GPU 

CPU2J 8*
Tout
2*
_output_shapes
: ??
?
e
,__inference_dropout_77_layer_call_fn_4793131

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-4792826*
Tout
2*P
fKRI
G__inference_dropout_77_layer_call_and_return_conditional_losses_4792815**
config_proto

GPU 

CPU2J 8*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?"
?
J__inference_sequential_79_layer_call_and_return_conditional_losses_4793032

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity??%hidden_layer_0/BiasAdd/ReadVariableOp?$hidden_layer_0/MatMul/ReadVariableOp?%hidden_layer_1/BiasAdd/ReadVariableOp?$hidden_layer_1/MatMul/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOpj
input_layer/Reshape/shapeConst*
valueB"????  *
_output_shapes
:*
dtype0}
input_layer/ReshapeReshapeinputs"input_layer/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
hidden_layer_0/MatMulMatMulinput_layer/Reshape:output:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_1/MatMulMatMul!hidden_layer_0/Relu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????u
dropout_77/IdentityIdentity!hidden_layer_1/Relu:activations:0*(
_output_shapes
:??????????*
T0?
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?
?
output_layer/MatMulMatMuldropout_77/Identity:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : 
?	
?
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_4792778

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792932

inputs1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinputs*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_4792726*
Tout
2*(
_output_shapes
:??????????*
Tin
2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-4792732?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_4792750*
Tin
2*(
_output_shapes
:??????????*
Tout
2*.
_gradient_op_typePartitionedCall-4792756**
config_proto

GPU 

CPU2J 8?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-4792784*
Tout
2*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_4792778*
Tin
2?
dropout_77/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*P
fKRI
G__inference_dropout_77_layer_call_and_return_conditional_losses_4792822*.
_gradient_op_typePartitionedCall-4792834*(
_output_shapes
:???????????
$output_layer/StatefulPartitionedCallStatefulPartitionedCall#dropout_77/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_4792850**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-4792856*
Tout
2*'
_output_shapes
:?????????
*
Tin
2?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
?
?
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792903

inputs1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??"dropout_77/StatefulPartitionedCall?&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-4792732*
Tin
2*
Tout
2*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_4792726**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:???????????
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*(
_output_shapes
:??????????*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_4792750*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*.
_gradient_op_typePartitionedCall-4792756?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-4792784*(
_output_shapes
:??????????*
Tin
2*
Tout
2*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_4792778?
"dropout_77/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*
Tin
2*(
_output_shapes
:??????????*
Tout
2**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_dropout_77_layer_call_and_return_conditional_losses_4792815*.
_gradient_op_typePartitionedCall-4792826?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall+dropout_77/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*
Tout
2*.
_gradient_op_typePartitionedCall-4792856*'
_output_shapes
:?????????
*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_4792850?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0#^dropout_77/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2H
"dropout_77/StatefulPartitionedCall"dropout_77/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
?
e
G__inference_dropout_77_layer_call_and_return_conditional_losses_4792822

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:??????????*
T0\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?	
?
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_4793094

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
/__inference_sequential_79_layer_call_fn_4793054

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tout
2*
Tin
	2*S
fNRL
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792932*'
_output_shapes
:?????????
*.
_gradient_op_typePartitionedCall-4792933**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
?
?
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792885
input_layer_input1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinput_layer_input*
Tin
2*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_4792726*.
_gradient_op_typePartitionedCall-4792732*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*
Tout
2?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_4792750*.
_gradient_op_typePartitionedCall-4792756*
Tout
2*
Tin
2?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*
Tout
2*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_4792778*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-4792784*
Tin
2**
config_proto

GPU 

CPU2J 8?
dropout_77/PartitionedCallPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*
Tout
2*P
fKRI
G__inference_dropout_77_layer_call_and_return_conditional_losses_4792822**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-4792834*
Tin
2*(
_output_shapes
:???????????
$output_layer/StatefulPartitionedCallStatefulPartitionedCall#dropout_77/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-4792856*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_4792850*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????
?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall: : : :1 -
+
_user_specified_nameinput_layer_input: : : 
?
?
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792868
input_layer_input1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??"dropout_77/StatefulPartitionedCall?&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinput_layer_input*
Tout
2*
Tin
2*.
_gradient_op_typePartitionedCall-4792732**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_4792726?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*(
_output_shapes
:??????????*
Tout
2*.
_gradient_op_typePartitionedCall-4792756*
Tin
2**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_4792750?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-4792784*
Tin
2*
Tout
2*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_4792778*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8?
"dropout_77/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-4792826*
Tin
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*
Tout
2*P
fKRI
G__inference_dropout_77_layer_call_and_return_conditional_losses_4792815?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall+dropout_77/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tout
2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_4792850*
Tin
2*.
_gradient_op_typePartitionedCall-4792856*'
_output_shapes
:?????????
?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0#^dropout_77/StatefulPartitionedCall'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2H
"dropout_77/StatefulPartitionedCall"dropout_77/StatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : 
?
?
.__inference_output_layer_layer_call_fn_4793154

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_4792850**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-4792856*'
_output_shapes
:?????????
*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
f
G__inference_dropout_77_layer_call_and_return_conditional_losses_4792815

inputs
identity?Q
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *ur?=C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*(
_output_shapes
:??????????*
dtype0*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*(
_output_shapes
:??????????*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*(
_output_shapes
:??????????*
T0R
dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????b
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:??????????*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:??????????*

SrcT0
j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?9
?
 __inference__traced_save_4793254
file_prefix7
3savev2_hidden_layer_0_79_kernel_read_readvariableop5
1savev2_hidden_layer_0_79_bias_read_readvariableop7
3savev2_hidden_layer_1_78_kernel_read_readvariableop5
1savev2_hidden_layer_1_78_bias_read_readvariableop5
1savev2_output_layer_79_kernel_read_readvariableop3
/savev2_output_layer_79_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop>
:savev2_adam_hidden_layer_0_79_kernel_m_read_readvariableop<
8savev2_adam_hidden_layer_0_79_bias_m_read_readvariableop>
:savev2_adam_hidden_layer_1_78_kernel_m_read_readvariableop<
8savev2_adam_hidden_layer_1_78_bias_m_read_readvariableop<
8savev2_adam_output_layer_79_kernel_m_read_readvariableop:
6savev2_adam_output_layer_79_bias_m_read_readvariableop>
:savev2_adam_hidden_layer_0_79_kernel_v_read_readvariableop<
8savev2_adam_hidden_layer_0_79_bias_v_read_readvariableop>
:savev2_adam_hidden_layer_1_78_kernel_v_read_readvariableop<
8savev2_adam_hidden_layer_1_78_bias_v_read_readvariableop<
8savev2_adam_output_layer_79_kernel_v_read_readvariableop:
6savev2_adam_output_layer_79_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_0fe657f1b11d4c28a55f6dc90a8dcd2c/part*
_output_shapes
: *
dtype0s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
value	B :*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_hidden_layer_0_79_kernel_read_readvariableop1savev2_hidden_layer_0_79_bias_read_readvariableop3savev2_hidden_layer_1_78_kernel_read_readvariableop1savev2_hidden_layer_1_78_bias_read_readvariableop1savev2_output_layer_79_kernel_read_readvariableop/savev2_output_layer_79_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop:savev2_adam_hidden_layer_0_79_kernel_m_read_readvariableop8savev2_adam_hidden_layer_0_79_bias_m_read_readvariableop:savev2_adam_hidden_layer_1_78_kernel_m_read_readvariableop8savev2_adam_hidden_layer_1_78_bias_m_read_readvariableop8savev2_adam_output_layer_79_kernel_m_read_readvariableop6savev2_adam_output_layer_79_bias_m_read_readvariableop:savev2_adam_hidden_layer_0_79_kernel_v_read_readvariableop8savev2_adam_hidden_layer_0_79_bias_v_read_readvariableop:savev2_adam_hidden_layer_1_78_kernel_v_read_readvariableop8savev2_adam_hidden_layer_1_78_bias_v_read_readvariableop8savev2_adam_output_layer_79_kernel_v_read_readvariableop6savev2_adam_output_layer_79_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *'
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:*
dtype0q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueB
B *
_output_shapes
:?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
N?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:	?
:
: : : : : : : :
??:?:
??:?:	?
:
:
??:?:
??:?:	?
:
: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
?
?
0__inference_hidden_layer_1_layer_call_fn_4793101

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_4792778*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-4792784*
Tin
2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?	
?
/__inference_sequential_79_layer_call_fn_4793043

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-4792904*
Tout
2*S
fNRL
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792903*'
_output_shapes
:?????????
*
Tin
	2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : 
?	
?
/__inference_sequential_79_layer_call_fn_4792942
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*S
fNRL
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792932*'
_output_shapes
:?????????
*
Tout
2*.
_gradient_op_typePartitionedCall-4792933*
Tin
	2**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: :1 -
+
_user_specified_nameinput_layer_input: : : : : 
?	
?
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_4793076

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
??j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*(
_output_shapes
:??????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?d
?
#__inference__traced_restore_4793342
file_prefix-
)assignvariableop_hidden_layer_0_79_kernel-
)assignvariableop_1_hidden_layer_0_79_bias/
+assignvariableop_2_hidden_layer_1_78_kernel-
)assignvariableop_3_hidden_layer_1_78_bias-
)assignvariableop_4_output_layer_79_kernel+
'assignvariableop_5_output_layer_79_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count7
3assignvariableop_13_adam_hidden_layer_0_79_kernel_m5
1assignvariableop_14_adam_hidden_layer_0_79_bias_m7
3assignvariableop_15_adam_hidden_layer_1_78_kernel_m5
1assignvariableop_16_adam_hidden_layer_1_78_bias_m5
1assignvariableop_17_adam_output_layer_79_kernel_m3
/assignvariableop_18_adam_output_layer_79_bias_m7
3assignvariableop_19_adam_hidden_layer_0_79_kernel_v5
1assignvariableop_20_adam_hidden_layer_0_79_bias_v7
3assignvariableop_21_adam_hidden_layer_1_78_kernel_v5
1assignvariableop_22_adam_hidden_layer_1_78_bias_v5
1assignvariableop_23_adam_output_layer_79_kernel_v3
/assignvariableop_24_adam_output_layer_79_bias_v
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:*
dtype0?
RestoreV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*'
dtypes
2	*x
_output_shapesf
d:::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0?
AssignVariableOpAssignVariableOp)assignvariableop_hidden_layer_0_79_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_hidden_layer_0_79_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp+assignvariableop_2_hidden_layer_1_78_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0?
AssignVariableOp_3AssignVariableOp)assignvariableop_3_hidden_layer_1_78_biasIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0?
AssignVariableOp_4AssignVariableOp)assignvariableop_4_output_layer_79_kernelIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0?
AssignVariableOp_5AssignVariableOp'assignvariableop_5_output_layer_79_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0	|
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:~
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:~
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:}
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0{
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0{
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp3assignvariableop_13_adam_hidden_layer_0_79_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype0P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp1assignvariableop_14_adam_hidden_layer_0_79_bias_mIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0?
AssignVariableOp_15AssignVariableOp3assignvariableop_15_adam_hidden_layer_1_78_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype0P
Identity_16IdentityRestoreV2:tensors:16*
_output_shapes
:*
T0?
AssignVariableOp_16AssignVariableOp1assignvariableop_16_adam_hidden_layer_1_78_bias_mIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0?
AssignVariableOp_17AssignVariableOp1assignvariableop_17_adam_output_layer_79_kernel_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0?
AssignVariableOp_18AssignVariableOp/assignvariableop_18_adam_output_layer_79_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0?
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_hidden_layer_0_79_kernel_vIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_hidden_layer_0_79_bias_vIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp3assignvariableop_21_adam_hidden_layer_1_78_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype0P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp1assignvariableop_22_adam_hidden_layer_1_78_bias_vIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp1assignvariableop_23_adam_output_layer_79_kernel_vIdentity_23:output:0*
_output_shapes
 *
dtype0P
Identity_24IdentityRestoreV2:tensors:24*
_output_shapes
:*
T0?
AssignVariableOp_24AssignVariableOp/assignvariableop_24_adam_output_layer_79_bias_vIdentity_24:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
_output_shapes
:*
dtype0?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_20:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : 
?	
?
%__inference_signature_wrapper_4792959
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*'
_output_shapes
:?????????
**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__wrapped_model_4792714*.
_gradient_op_typePartitionedCall-4792950*
Tout
2*
Tin
	2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :1 -
+
_user_specified_nameinput_layer_input: 
?3
?
J__inference_sequential_79_layer_call_and_return_conditional_losses_4793004

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity??%hidden_layer_0/BiasAdd/ReadVariableOp?$hidden_layer_0/MatMul/ReadVariableOp?%hidden_layer_1/BiasAdd/ReadVariableOp?$hidden_layer_1/MatMul/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOpj
input_layer/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"????  }
input_layer/ReshapeReshapeinputs"input_layer/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0?
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
hidden_layer_0/MatMulMatMulinput_layer/Reshape:output:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_1/MatMulMatMul!hidden_layer_0/Relu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????\
dropout_77/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *ur?=i
dropout_77/dropout/ShapeShape!hidden_layer_1/Relu:activations:0*
_output_shapes
:*
T0j
%dropout_77/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    j
%dropout_77/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
/dropout_77/dropout/random_uniform/RandomUniformRandomUniform!dropout_77/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:???????????
%dropout_77/dropout/random_uniform/subSub.dropout_77/dropout/random_uniform/max:output:0.dropout_77/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
%dropout_77/dropout/random_uniform/mulMul8dropout_77/dropout/random_uniform/RandomUniform:output:0)dropout_77/dropout/random_uniform/sub:z:0*(
_output_shapes
:??????????*
T0?
!dropout_77/dropout/random_uniformAdd)dropout_77/dropout/random_uniform/mul:z:0.dropout_77/dropout/random_uniform/min:output:0*(
_output_shapes
:??????????*
T0]
dropout_77/dropout/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0?
dropout_77/dropout/subSub!dropout_77/dropout/sub/x:output:0 dropout_77/dropout/rate:output:0*
_output_shapes
: *
T0a
dropout_77/dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
dropout_77/dropout/truedivRealDiv%dropout_77/dropout/truediv/x:output:0dropout_77/dropout/sub:z:0*
_output_shapes
: *
T0?
dropout_77/dropout/GreaterEqualGreaterEqual%dropout_77/dropout/random_uniform:z:0 dropout_77/dropout/rate:output:0*(
_output_shapes
:??????????*
T0?
dropout_77/dropout/mulMul!hidden_layer_1/Relu:activations:0dropout_77/dropout/truediv:z:0*(
_output_shapes
:??????????*
T0?
dropout_77/dropout/CastCast#dropout_77/dropout/GreaterEqual:z:0*(
_output_shapes
:??????????*

DstT0*

SrcT0
?
dropout_77/dropout/mul_1Muldropout_77/dropout/mul:z:0dropout_77/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?
?
output_layer/MatMulMatMuldropout_77/dropout/mul_1:z:0*output_layer/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : 
?
d
H__inference_input_layer_layer_call_and_return_conditional_losses_4793060

inputs
identity^
Reshape/shapeConst*
valueB"????  *
_output_shapes
:*
dtype0e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?	
?
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_4792750

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
d
H__inference_input_layer_layer_call_and_return_conditional_losses_4792726

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
valueB"????  *
dtype0e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?	
?
I__inference_output_layer_layer_call_and_return_conditional_losses_4792850

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?
i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
f
G__inference_dropout_77_layer_call_and_return_conditional_losses_4793121

inputs
identity?Q
dropout/rateConst*
_output_shapes
: *
valueB
 *ur?=*
dtype0C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    _
dropout/random_uniform/maxConst*
dtype0*
valueB
 *  ??*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*(
_output_shapes
:??????????*
dtype0*
T0?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*(
_output_shapes
:??????????*
T0?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????R
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:??????????*
T0b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*(
_output_shapes
:??????????*

DstT0*

SrcT0
j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/mul_1:z:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?*
?
"__inference__wrapped_model_4792714
input_layer_input?
;sequential_79_hidden_layer_0_matmul_readvariableop_resource@
<sequential_79_hidden_layer_0_biasadd_readvariableop_resource?
;sequential_79_hidden_layer_1_matmul_readvariableop_resource@
<sequential_79_hidden_layer_1_biasadd_readvariableop_resource=
9sequential_79_output_layer_matmul_readvariableop_resource>
:sequential_79_output_layer_biasadd_readvariableop_resource
identity??3sequential_79/hidden_layer_0/BiasAdd/ReadVariableOp?2sequential_79/hidden_layer_0/MatMul/ReadVariableOp?3sequential_79/hidden_layer_1/BiasAdd/ReadVariableOp?2sequential_79/hidden_layer_1/MatMul/ReadVariableOp?1sequential_79/output_layer/BiasAdd/ReadVariableOp?0sequential_79/output_layer/MatMul/ReadVariableOpx
'sequential_79/input_layer/Reshape/shapeConst*
valueB"????  *
dtype0*
_output_shapes
:?
!sequential_79/input_layer/ReshapeReshapeinput_layer_input0sequential_79/input_layer/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0?
2sequential_79/hidden_layer_0/MatMul/ReadVariableOpReadVariableOp;sequential_79_hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
#sequential_79/hidden_layer_0/MatMulMatMul*sequential_79/input_layer/Reshape:output:0:sequential_79/hidden_layer_0/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
3sequential_79/hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp<sequential_79_hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
$sequential_79/hidden_layer_0/BiasAddBiasAdd-sequential_79/hidden_layer_0/MatMul:product:0;sequential_79/hidden_layer_0/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
!sequential_79/hidden_layer_0/ReluRelu-sequential_79/hidden_layer_0/BiasAdd:output:0*(
_output_shapes
:??????????*
T0?
2sequential_79/hidden_layer_1/MatMul/ReadVariableOpReadVariableOp;sequential_79_hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
#sequential_79/hidden_layer_1/MatMulMatMul/sequential_79/hidden_layer_0/Relu:activations:0:sequential_79/hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
3sequential_79/hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp<sequential_79_hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
$sequential_79/hidden_layer_1/BiasAddBiasAdd-sequential_79/hidden_layer_1/MatMul:product:0;sequential_79/hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!sequential_79/hidden_layer_1/ReluRelu-sequential_79/hidden_layer_1/BiasAdd:output:0*(
_output_shapes
:??????????*
T0?
!sequential_79/dropout_77/IdentityIdentity/sequential_79/hidden_layer_1/Relu:activations:0*
T0*(
_output_shapes
:???????????
0sequential_79/output_layer/MatMul/ReadVariableOpReadVariableOp9sequential_79_output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?
?
!sequential_79/output_layer/MatMulMatMul*sequential_79/dropout_77/Identity:output:08sequential_79/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
1sequential_79/output_layer/BiasAdd/ReadVariableOpReadVariableOp:sequential_79_output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
"sequential_79/output_layer/BiasAddBiasAdd+sequential_79/output_layer/MatMul:product:09sequential_79/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
"sequential_79/output_layer/SoftmaxSoftmax+sequential_79/output_layer/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentity,sequential_79/output_layer/Softmax:softmax:04^sequential_79/hidden_layer_0/BiasAdd/ReadVariableOp3^sequential_79/hidden_layer_0/MatMul/ReadVariableOp4^sequential_79/hidden_layer_1/BiasAdd/ReadVariableOp3^sequential_79/hidden_layer_1/MatMul/ReadVariableOp2^sequential_79/output_layer/BiasAdd/ReadVariableOp1^sequential_79/output_layer/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::2h
2sequential_79/hidden_layer_1/MatMul/ReadVariableOp2sequential_79/hidden_layer_1/MatMul/ReadVariableOp2h
2sequential_79/hidden_layer_0/MatMul/ReadVariableOp2sequential_79/hidden_layer_0/MatMul/ReadVariableOp2j
3sequential_79/hidden_layer_1/BiasAdd/ReadVariableOp3sequential_79/hidden_layer_1/BiasAdd/ReadVariableOp2f
1sequential_79/output_layer/BiasAdd/ReadVariableOp1sequential_79/output_layer/BiasAdd/ReadVariableOp2j
3sequential_79/hidden_layer_0/BiasAdd/ReadVariableOp3sequential_79/hidden_layer_0/BiasAdd/ReadVariableOp2d
0sequential_79/output_layer/MatMul/ReadVariableOp0sequential_79/output_layer/MatMul/ReadVariableOp:1 -
+
_user_specified_nameinput_layer_input: : : : : : 
?	
?
I__inference_output_layer_layer_call_and_return_conditional_losses_4793147

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
0__inference_hidden_layer_0_layer_call_fn_4793083

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*(
_output_shapes
:??????????*
Tout
2*
Tin
2*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_4792750**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-4792756?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?	
?
/__inference_sequential_79_layer_call_fn_4792913
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tout
2*
Tin
	2*.
_gradient_op_typePartitionedCall-4792904*S
fNRL
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792903**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*B
_input_shapes1
/:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : 
?
I
-__inference_input_layer_layer_call_fn_4793065

inputs
identity?
PartitionedCallPartitionedCallinputs*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_4792726**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*
Tout
2*.
_gradient_op_typePartitionedCall-4792732*
Tin
2a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
e
G__inference_dropout_77_layer_call_and_return_conditional_losses_4793126

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:??????????*
T0\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
H
,__inference_dropout_77_layer_call_fn_4793136

inputs
identity?
PartitionedCallPartitionedCallinputs*P
fKRI
G__inference_dropout_77_layer_call_and_return_conditional_losses_4792822*.
_gradient_op_typePartitionedCall-4792834*
Tout
2*(
_output_shapes
:??????????*
Tin
2**
config_proto

GPU 

CPU2J 8a
IdentityIdentityPartitionedCall:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
S
input_layer_input>
#serving_default_input_layer_input:0?????????@
output_layer0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
?#
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
	optimizer
	variables
	regularization_losses

trainable_variables
	keras_api

signatures
d__call__
*e&call_and_return_all_conditional_losses
f_default_save_signature"? 
_tf_keras_sequential? {"class_name": "Sequential", "name": "sequential_79", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_79", "layers": [{"class_name": "Flatten", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 698, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 470, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_77", "trainable": true, "dtype": "float32", "rate": 0.09933940306516521, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_79", "layers": [{"class_name": "Flatten", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 698, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 470, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_77", "trainable": true, "dtype": "float32", "rate": 0.09933940306516521, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
g__call__
*h&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "input_layer_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28, 28], "config": {"batch_input_shape": [null, 28, 28], "dtype": "float32", "sparse": false, "name": "input_layer_input"}}
?
	variables
regularization_losses
trainable_variables
	keras_api
i__call__
*j&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "input_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 28, 28], "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
k__call__
*l&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 698, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
m__call__
*n&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 470, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 698}}}}
?
!	variables
"regularization_losses
#trainable_variables
$	keras_api
o__call__
*p&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_77", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_77", "trainable": true, "dtype": "float32", "rate": 0.09933940306516521, "noise_shape": null, "seed": null}}
?

%kernel
&bias
'	variables
(regularization_losses
)trainable_variables
*	keras_api
q__call__
*r&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 470}}}}
?
+iter

,beta_1

-beta_2
	.decay
/learning_ratemXmYmZm[%m\&m]v^v_v`va%vb&vc"
	optimizer
J
0
1
2
3
%4
&5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
%4
&5"
trackable_list_wrapper
?
	variables
	regularization_losses

trainable_variables

0layers
1metrics
2non_trainable_variables
3layer_regularization_losses
d__call__
f_default_save_signature
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
,
sserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses
trainable_variables

4layers
5metrics
6non_trainable_variables
7layer_regularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses
trainable_variables

8layers
9metrics
:non_trainable_variables
;layer_regularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
,:*
??2hidden_layer_0_79/kernel
%:#?2hidden_layer_0_79/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
regularization_losses
trainable_variables

<layers
=metrics
>non_trainable_variables
?layer_regularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,:*
??2hidden_layer_1_78/kernel
%:#?2hidden_layer_1_78/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
regularization_losses
trainable_variables

@layers
Ametrics
Bnon_trainable_variables
Clayer_regularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
!	variables
"regularization_losses
#trainable_variables

Dlayers
Emetrics
Fnon_trainable_variables
Glayer_regularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
):'	?
2output_layer_79/kernel
": 
2output_layer_79/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
'	variables
(regularization_losses
)trainable_variables

Hlayers
Imetrics
Jnon_trainable_variables
Klayer_regularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
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
'
L0"
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
 "
trackable_list_wrapper
?
	Mtotal
	Ncount
O
_fn_kwargs
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
t__call__
*u&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
P	variables
Qregularization_losses
Rtrainable_variables

Tlayers
Umetrics
Vnon_trainable_variables
Wlayer_regularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
1:/
??2Adam/hidden_layer_0_79/kernel/m
*:(?2Adam/hidden_layer_0_79/bias/m
1:/
??2Adam/hidden_layer_1_78/kernel/m
*:(?2Adam/hidden_layer_1_78/bias/m
.:,	?
2Adam/output_layer_79/kernel/m
':%
2Adam/output_layer_79/bias/m
1:/
??2Adam/hidden_layer_0_79/kernel/v
*:(?2Adam/hidden_layer_0_79/bias/v
1:/
??2Adam/hidden_layer_1_78/kernel/v
*:(?2Adam/hidden_layer_1_78/bias/v
.:,	?
2Adam/output_layer_79/kernel/v
':%
2Adam/output_layer_79/bias/v
?2?
/__inference_sequential_79_layer_call_fn_4792942
/__inference_sequential_79_layer_call_fn_4793054
/__inference_sequential_79_layer_call_fn_4793043
/__inference_sequential_79_layer_call_fn_4792913?
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
?2?
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792868
J__inference_sequential_79_layer_call_and_return_conditional_losses_4793004
J__inference_sequential_79_layer_call_and_return_conditional_losses_4793032
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792885?
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
"__inference__wrapped_model_4792714?
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
annotations? *4?1
/?,
input_layer_input?????????
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
-__inference_input_layer_layer_call_fn_4793065?
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
H__inference_input_layer_layer_call_and_return_conditional_losses_4793060?
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
0__inference_hidden_layer_0_layer_call_fn_4793083?
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
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_4793076?
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
0__inference_hidden_layer_1_layer_call_fn_4793101?
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
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_4793094?
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
?2?
,__inference_dropout_77_layer_call_fn_4793136
,__inference_dropout_77_layer_call_fn_4793131?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_77_layer_call_and_return_conditional_losses_4793126
G__inference_dropout_77_layer_call_and_return_conditional_losses_4793121?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_output_layer_layer_call_fn_4793154?
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
I__inference_output_layer_layer_call_and_return_conditional_losses_4793147?
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
>B<
%__inference_signature_wrapper_4792959input_layer_input
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
0__inference_hidden_layer_0_layer_call_fn_4793083Q0?-
&?#
!?
inputs??????????
? "????????????
/__inference_sequential_79_layer_call_fn_4792913j%&F?C
<?9
/?,
input_layer_input?????????
p

 
? "??????????
?
J__inference_sequential_79_layer_call_and_return_conditional_losses_4793004l%&;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????

? ?
.__inference_output_layer_layer_call_fn_4793154P%&0?-
&?#
!?
inputs??????????
? "??????????
?
"__inference__wrapped_model_4792714?%&>?;
4?1
/?,
input_layer_input?????????
? ";?8
6
output_layer&?#
output_layer?????????
?
/__inference_sequential_79_layer_call_fn_4793054_%&;?8
1?.
$?!
inputs?????????
p 

 
? "??????????
?
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_4793076^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
I__inference_output_layer_layer_call_and_return_conditional_losses_4793147]%&0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? ?
%__inference_signature_wrapper_4792959?%&S?P
? 
I?F
D
input_layer_input/?,
input_layer_input?????????";?8
6
output_layer&?#
output_layer?????????
?
0__inference_hidden_layer_1_layer_call_fn_4793101Q0?-
&?#
!?
inputs??????????
? "????????????
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792868w%&F?C
<?9
/?,
input_layer_input?????????
p

 
? "%?"
?
0?????????

? ?
J__inference_sequential_79_layer_call_and_return_conditional_losses_4793032l%&;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
-__inference_input_layer_layer_call_fn_4793065P3?0
)?&
$?!
inputs?????????
? "????????????
/__inference_sequential_79_layer_call_fn_4792942j%&F?C
<?9
/?,
input_layer_input?????????
p 

 
? "??????????
?
G__inference_dropout_77_layer_call_and_return_conditional_losses_4793126^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
G__inference_dropout_77_layer_call_and_return_conditional_losses_4793121^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
H__inference_input_layer_layer_call_and_return_conditional_losses_4793060]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? ?
,__inference_dropout_77_layer_call_fn_4793136Q4?1
*?'
!?
inputs??????????
p 
? "????????????
/__inference_sequential_79_layer_call_fn_4793043_%&;?8
1?.
$?!
inputs?????????
p

 
? "??????????
?
,__inference_dropout_77_layer_call_fn_4793131Q4?1
*?'
!?
inputs??????????
p
? "????????????
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_4793094^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
J__inference_sequential_79_layer_call_and_return_conditional_losses_4792885w%&F?C
<?9
/?,
input_layer_input?????????
p 

 
? "%?"
?
0?????????

? 