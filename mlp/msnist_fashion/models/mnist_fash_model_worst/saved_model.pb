??	
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
shapeshape?"serve*2.0.02unknown8??
?
hidden_layer_0/kernelVarHandleOp*
dtype0*
shape:
??*
_output_shapes
: *&
shared_namehidden_layer_0/kernel
?
)hidden_layer_0/kernel/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel*
dtype0* 
_output_shapes
:
??

hidden_layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*$
shared_namehidden_layer_0/bias
x
'hidden_layer_0/bias/Read/ReadVariableOpReadVariableOphidden_layer_0/bias*
dtype0*
_output_shapes	
:?
?
hidden_layer_1/kernelVarHandleOp*
_output_shapes
: *
shape:
??*
dtype0*&
shared_namehidden_layer_1/kernel
?
)hidden_layer_1/kernel/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel*
dtype0* 
_output_shapes
:
??

hidden_layer_1/biasVarHandleOp*
shape:?*
_output_shapes
: *$
shared_namehidden_layer_1/bias*
dtype0
x
'hidden_layer_1/bias/Read/ReadVariableOpReadVariableOphidden_layer_1/bias*
dtype0*
_output_shapes	
:?
?
hidden_layer_2/kernelVarHandleOp*
shape:
??*
dtype0*&
shared_namehidden_layer_2/kernel*
_output_shapes
: 
?
)hidden_layer_2/kernel/Read/ReadVariableOpReadVariableOphidden_layer_2/kernel*
dtype0* 
_output_shapes
:
??

hidden_layer_2/biasVarHandleOp*$
shared_namehidden_layer_2/bias*
_output_shapes
: *
dtype0*
shape:?
x
'hidden_layer_2/bias/Read/ReadVariableOpReadVariableOphidden_layer_2/bias*
_output_shapes	
:?*
dtype0
?
hidden_layer_3/kernelVarHandleOp*
_output_shapes
: *&
shared_namehidden_layer_3/kernel*
shape:
??*
dtype0
?
)hidden_layer_3/kernel/Read/ReadVariableOpReadVariableOphidden_layer_3/kernel*
dtype0* 
_output_shapes
:
??

hidden_layer_3/biasVarHandleOp*$
shared_namehidden_layer_3/bias*
_output_shapes
: *
dtype0*
shape:?
x
'hidden_layer_3/bias/Read/ReadVariableOpReadVariableOphidden_layer_3/bias*
dtype0*
_output_shapes	
:?
?
hidden_layer_4/kernelVarHandleOp*&
shared_namehidden_layer_4/kernel*
_output_shapes
: *
shape:
??*
dtype0
?
)hidden_layer_4/kernel/Read/ReadVariableOpReadVariableOphidden_layer_4/kernel* 
_output_shapes
:
??*
dtype0

hidden_layer_4/biasVarHandleOp*$
shared_namehidden_layer_4/bias*
_output_shapes
: *
shape:?*
dtype0
x
'hidden_layer_4/bias/Read/ReadVariableOpReadVariableOphidden_layer_4/bias*
_output_shapes	
:?*
dtype0
?
output_layer/kernelVarHandleOp*
shape:	?
*$
shared_nameoutput_layer/kernel*
_output_shapes
: *
dtype0
|
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes
:	?
*
dtype0
z
output_layer/biasVarHandleOp*
dtype0*"
shared_nameoutput_layer/bias*
_output_shapes
: *
shape:

s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
dtype0*
_output_shapes
:

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
countVarHandleOp*
_output_shapes
: *
shared_namecount*
shape: *
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
hidden_layer_0/kernel/mVarHandleOp*
_output_shapes
: *(
shared_namehidden_layer_0/kernel/m*
dtype0*
shape:
??
?
+hidden_layer_0/kernel/m/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel/m*
dtype0* 
_output_shapes
:
??
?
hidden_layer_0/bias/mVarHandleOp*&
shared_namehidden_layer_0/bias/m*
shape:?*
dtype0*
_output_shapes
: 
|
)hidden_layer_0/bias/m/Read/ReadVariableOpReadVariableOphidden_layer_0/bias/m*
dtype0*
_output_shapes	
:?
?
hidden_layer_1/kernel/mVarHandleOp*
shape:
??*(
shared_namehidden_layer_1/kernel/m*
dtype0*
_output_shapes
: 
?
+hidden_layer_1/kernel/m/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel/m*
dtype0* 
_output_shapes
:
??
?
hidden_layer_1/bias/mVarHandleOp*&
shared_namehidden_layer_1/bias/m*
shape:?*
dtype0*
_output_shapes
: 
|
)hidden_layer_1/bias/m/Read/ReadVariableOpReadVariableOphidden_layer_1/bias/m*
_output_shapes	
:?*
dtype0
?
hidden_layer_2/kernel/mVarHandleOp*
dtype0*(
shared_namehidden_layer_2/kernel/m*
_output_shapes
: *
shape:
??
?
+hidden_layer_2/kernel/m/Read/ReadVariableOpReadVariableOphidden_layer_2/kernel/m* 
_output_shapes
:
??*
dtype0
?
hidden_layer_2/bias/mVarHandleOp*
_output_shapes
: *
shape:?*
dtype0*&
shared_namehidden_layer_2/bias/m
|
)hidden_layer_2/bias/m/Read/ReadVariableOpReadVariableOphidden_layer_2/bias/m*
dtype0*
_output_shapes	
:?
?
hidden_layer_3/kernel/mVarHandleOp*(
shared_namehidden_layer_3/kernel/m*
dtype0*
shape:
??*
_output_shapes
: 
?
+hidden_layer_3/kernel/m/Read/ReadVariableOpReadVariableOphidden_layer_3/kernel/m* 
_output_shapes
:
??*
dtype0
?
hidden_layer_3/bias/mVarHandleOp*
shape:?*
dtype0*&
shared_namehidden_layer_3/bias/m*
_output_shapes
: 
|
)hidden_layer_3/bias/m/Read/ReadVariableOpReadVariableOphidden_layer_3/bias/m*
dtype0*
_output_shapes	
:?
?
hidden_layer_4/kernel/mVarHandleOp*(
shared_namehidden_layer_4/kernel/m*
dtype0*
shape:
??*
_output_shapes
: 
?
+hidden_layer_4/kernel/m/Read/ReadVariableOpReadVariableOphidden_layer_4/kernel/m*
dtype0* 
_output_shapes
:
??
?
hidden_layer_4/bias/mVarHandleOp*
_output_shapes
: *&
shared_namehidden_layer_4/bias/m*
shape:?*
dtype0
|
)hidden_layer_4/bias/m/Read/ReadVariableOpReadVariableOphidden_layer_4/bias/m*
_output_shapes	
:?*
dtype0
?
output_layer/kernel/mVarHandleOp*
_output_shapes
: *&
shared_nameoutput_layer/kernel/m*
dtype0*
shape:	?

?
)output_layer/kernel/m/Read/ReadVariableOpReadVariableOpoutput_layer/kernel/m*
dtype0*
_output_shapes
:	?

~
output_layer/bias/mVarHandleOp*
_output_shapes
: *$
shared_nameoutput_layer/bias/m*
dtype0*
shape:

w
'output_layer/bias/m/Read/ReadVariableOpReadVariableOpoutput_layer/bias/m*
_output_shapes
:
*
dtype0
?
hidden_layer_0/kernel/vVarHandleOp*(
shared_namehidden_layer_0/kernel/v*
shape:
??*
_output_shapes
: *
dtype0
?
+hidden_layer_0/kernel/v/Read/ReadVariableOpReadVariableOphidden_layer_0/kernel/v* 
_output_shapes
:
??*
dtype0
?
hidden_layer_0/bias/vVarHandleOp*
dtype0*&
shared_namehidden_layer_0/bias/v*
_output_shapes
: *
shape:?
|
)hidden_layer_0/bias/v/Read/ReadVariableOpReadVariableOphidden_layer_0/bias/v*
_output_shapes	
:?*
dtype0
?
hidden_layer_1/kernel/vVarHandleOp*
_output_shapes
: *(
shared_namehidden_layer_1/kernel/v*
shape:
??*
dtype0
?
+hidden_layer_1/kernel/v/Read/ReadVariableOpReadVariableOphidden_layer_1/kernel/v*
dtype0* 
_output_shapes
:
??
?
hidden_layer_1/bias/vVarHandleOp*
shape:?*
dtype0*
_output_shapes
: *&
shared_namehidden_layer_1/bias/v
|
)hidden_layer_1/bias/v/Read/ReadVariableOpReadVariableOphidden_layer_1/bias/v*
dtype0*
_output_shapes	
:?
?
hidden_layer_2/kernel/vVarHandleOp*
shape:
??*(
shared_namehidden_layer_2/kernel/v*
dtype0*
_output_shapes
: 
?
+hidden_layer_2/kernel/v/Read/ReadVariableOpReadVariableOphidden_layer_2/kernel/v*
dtype0* 
_output_shapes
:
??
?
hidden_layer_2/bias/vVarHandleOp*
_output_shapes
: *&
shared_namehidden_layer_2/bias/v*
shape:?*
dtype0
|
)hidden_layer_2/bias/v/Read/ReadVariableOpReadVariableOphidden_layer_2/bias/v*
_output_shapes	
:?*
dtype0
?
hidden_layer_3/kernel/vVarHandleOp*
_output_shapes
: *
shape:
??*(
shared_namehidden_layer_3/kernel/v*
dtype0
?
+hidden_layer_3/kernel/v/Read/ReadVariableOpReadVariableOphidden_layer_3/kernel/v*
dtype0* 
_output_shapes
:
??
?
hidden_layer_3/bias/vVarHandleOp*&
shared_namehidden_layer_3/bias/v*
_output_shapes
: *
dtype0*
shape:?
|
)hidden_layer_3/bias/v/Read/ReadVariableOpReadVariableOphidden_layer_3/bias/v*
_output_shapes	
:?*
dtype0
?
hidden_layer_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*(
shared_namehidden_layer_4/kernel/v*
shape:
??
?
+hidden_layer_4/kernel/v/Read/ReadVariableOpReadVariableOphidden_layer_4/kernel/v* 
_output_shapes
:
??*
dtype0
?
hidden_layer_4/bias/vVarHandleOp*&
shared_namehidden_layer_4/bias/v*
_output_shapes
: *
shape:?*
dtype0
|
)hidden_layer_4/bias/v/Read/ReadVariableOpReadVariableOphidden_layer_4/bias/v*
dtype0*
_output_shapes	
:?
?
output_layer/kernel/vVarHandleOp*
_output_shapes
: *
shape:	?
*&
shared_nameoutput_layer/kernel/v*
dtype0
?
)output_layer/kernel/v/Read/ReadVariableOpReadVariableOpoutput_layer/kernel/v*
dtype0*
_output_shapes
:	?

~
output_layer/bias/vVarHandleOp*$
shared_nameoutput_layer/bias/v*
_output_shapes
: *
shape:
*
dtype0
w
'output_layer/bias/v/Read/ReadVariableOpReadVariableOpoutput_layer/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *?>
value?>B?> B?>
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		optimizer

	variables
regularization_losses
trainable_variables
	keras_api

signatures
R
	variables
regularization_losses
trainable_variables
	keras_api
R
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
h

/kernel
0bias
1	variables
2regularization_losses
3trainable_variables
4	keras_api
h

5kernel
6bias
7	variables
8regularization_losses
9trainable_variables
:	keras_api
?mkmlmmmn#mo$mp)mq*mr/ms0mt5mu6mvvwvxvyvz#v{$v|)v}*v~/v0v?5v?6v?
V
0
1
2
3
#4
$5
)6
*7
/8
09
510
611
 
V
0
1
2
3
#4
$5
)6
*7
/8
09
510
611
?

	variables
regularization_losses

;layers
trainable_variables
<non_trainable_variables
=layer_regularization_losses
>metrics
 
 
 
 
?
	variables
regularization_losses

?layers
trainable_variables
@metrics
Alayer_regularization_losses
Bnon_trainable_variables
 
 
 
?
	variables
regularization_losses

Clayers
trainable_variables
Dmetrics
Elayer_regularization_losses
Fnon_trainable_variables
a_
VARIABLE_VALUEhidden_layer_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
regularization_losses

Glayers
trainable_variables
Hmetrics
Ilayer_regularization_losses
Jnon_trainable_variables
a_
VARIABLE_VALUEhidden_layer_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
 regularization_losses

Klayers
!trainable_variables
Lmetrics
Mlayer_regularization_losses
Nnon_trainable_variables
a_
VARIABLE_VALUEhidden_layer_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
?
%	variables
&regularization_losses

Olayers
'trainable_variables
Pmetrics
Qlayer_regularization_losses
Rnon_trainable_variables
a_
VARIABLE_VALUEhidden_layer_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
?
+	variables
,regularization_losses

Slayers
-trainable_variables
Tmetrics
Ulayer_regularization_losses
Vnon_trainable_variables
a_
VARIABLE_VALUEhidden_layer_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEhidden_layer_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01
 

/0
01
?
1	variables
2regularization_losses

Wlayers
3trainable_variables
Xmetrics
Ylayer_regularization_losses
Znon_trainable_variables
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

50
61
 

50
61
?
7	variables
8regularization_losses

[layers
9trainable_variables
\metrics
]layer_regularization_losses
^non_trainable_variables
1
0
1
2
3
4
5
6
 
 

_0
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
 
 
 
 
 
 
x
	`total
	acount
b
_fn_kwargs
c	variables
dregularization_losses
etrainable_variables
f	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

`0
a1
 
 
?
c	variables
dregularization_losses

glayers
etrainable_variables
hmetrics
ilayer_regularization_losses
jnon_trainable_variables
 
 
 

`0
a1
}
VARIABLE_VALUEhidden_layer_0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEoutput_layer/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEoutput_layer/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEhidden_layer_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEhidden_layer_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEoutput_layer/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEoutput_layer/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
!serving_default_input_layer_inputPlaceholder* 
shape:?????????*+
_output_shapes
:?????????*
dtype0
?
StatefulPartitionedCallStatefulPartitionedCall!serving_default_input_layer_inputhidden_layer_0/kernelhidden_layer_0/biashidden_layer_1/kernelhidden_layer_1/biashidden_layer_2/kernelhidden_layer_2/biashidden_layer_3/kernelhidden_layer_3/biashidden_layer_4/kernelhidden_layer_4/biasoutput_layer/kerneloutput_layer/bias*'
_output_shapes
:?????????
**
config_proto

GPU 

CPU2J 8*.
f)R'
%__inference_signature_wrapper_6735283*
Tin
2*.
_gradient_op_typePartitionedCall-6735614*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)hidden_layer_0/kernel/Read/ReadVariableOp'hidden_layer_0/bias/Read/ReadVariableOp)hidden_layer_1/kernel/Read/ReadVariableOp'hidden_layer_1/bias/Read/ReadVariableOp)hidden_layer_2/kernel/Read/ReadVariableOp'hidden_layer_2/bias/Read/ReadVariableOp)hidden_layer_3/kernel/Read/ReadVariableOp'hidden_layer_3/bias/Read/ReadVariableOp)hidden_layer_4/kernel/Read/ReadVariableOp'hidden_layer_4/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+hidden_layer_0/kernel/m/Read/ReadVariableOp)hidden_layer_0/bias/m/Read/ReadVariableOp+hidden_layer_1/kernel/m/Read/ReadVariableOp)hidden_layer_1/bias/m/Read/ReadVariableOp+hidden_layer_2/kernel/m/Read/ReadVariableOp)hidden_layer_2/bias/m/Read/ReadVariableOp+hidden_layer_3/kernel/m/Read/ReadVariableOp)hidden_layer_3/bias/m/Read/ReadVariableOp+hidden_layer_4/kernel/m/Read/ReadVariableOp)hidden_layer_4/bias/m/Read/ReadVariableOp)output_layer/kernel/m/Read/ReadVariableOp'output_layer/bias/m/Read/ReadVariableOp+hidden_layer_0/kernel/v/Read/ReadVariableOp)hidden_layer_0/bias/v/Read/ReadVariableOp+hidden_layer_1/kernel/v/Read/ReadVariableOp)hidden_layer_1/bias/v/Read/ReadVariableOp+hidden_layer_2/kernel/v/Read/ReadVariableOp)hidden_layer_2/bias/v/Read/ReadVariableOp+hidden_layer_3/kernel/v/Read/ReadVariableOp)hidden_layer_3/bias/v/Read/ReadVariableOp+hidden_layer_4/kernel/v/Read/ReadVariableOp)hidden_layer_4/bias/v/Read/ReadVariableOp)output_layer/kernel/v/Read/ReadVariableOp'output_layer/bias/v/Read/ReadVariableOpConst*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_save_6735673*.
_gradient_op_typePartitionedCall-6735674*3
Tin,
*2(*
Tout
2
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_layer_0/kernelhidden_layer_0/biashidden_layer_1/kernelhidden_layer_1/biashidden_layer_2/kernelhidden_layer_2/biashidden_layer_3/kernelhidden_layer_3/biashidden_layer_4/kernelhidden_layer_4/biasoutput_layer/kerneloutput_layer/biastotalcounthidden_layer_0/kernel/mhidden_layer_0/bias/mhidden_layer_1/kernel/mhidden_layer_1/bias/mhidden_layer_2/kernel/mhidden_layer_2/bias/mhidden_layer_3/kernel/mhidden_layer_3/bias/mhidden_layer_4/kernel/mhidden_layer_4/bias/moutput_layer/kernel/moutput_layer/bias/mhidden_layer_0/kernel/vhidden_layer_0/bias/vhidden_layer_1/kernel/vhidden_layer_1/bias/vhidden_layer_2/kernel/vhidden_layer_2/bias/vhidden_layer_3/kernel/vhidden_layer_3/bias/vhidden_layer_4/kernel/vhidden_layer_4/bias/voutput_layer/kernel/voutput_layer/bias/v*
_output_shapes
: *.
_gradient_op_typePartitionedCall-6735801*
Tout
2*2
Tin+
)2'**
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference__traced_restore_6735800޹
??
?
#__inference__traced_restore_6735800
file_prefix*
&assignvariableop_hidden_layer_0_kernel*
&assignvariableop_1_hidden_layer_0_bias,
(assignvariableop_2_hidden_layer_1_kernel*
&assignvariableop_3_hidden_layer_1_bias,
(assignvariableop_4_hidden_layer_2_kernel*
&assignvariableop_5_hidden_layer_2_bias,
(assignvariableop_6_hidden_layer_3_kernel*
&assignvariableop_7_hidden_layer_3_bias,
(assignvariableop_8_hidden_layer_4_kernel*
&assignvariableop_9_hidden_layer_4_bias+
'assignvariableop_10_output_layer_kernel)
%assignvariableop_11_output_layer_bias
assignvariableop_12_total
assignvariableop_13_count/
+assignvariableop_14_hidden_layer_0_kernel_m-
)assignvariableop_15_hidden_layer_0_bias_m/
+assignvariableop_16_hidden_layer_1_kernel_m-
)assignvariableop_17_hidden_layer_1_bias_m/
+assignvariableop_18_hidden_layer_2_kernel_m-
)assignvariableop_19_hidden_layer_2_bias_m/
+assignvariableop_20_hidden_layer_3_kernel_m-
)assignvariableop_21_hidden_layer_3_bias_m/
+assignvariableop_22_hidden_layer_4_kernel_m-
)assignvariableop_23_hidden_layer_4_bias_m-
)assignvariableop_24_output_layer_kernel_m+
'assignvariableop_25_output_layer_bias_m/
+assignvariableop_26_hidden_layer_0_kernel_v-
)assignvariableop_27_hidden_layer_0_bias_v/
+assignvariableop_28_hidden_layer_1_kernel_v-
)assignvariableop_29_hidden_layer_1_bias_v/
+assignvariableop_30_hidden_layer_2_kernel_v-
)assignvariableop_31_hidden_layer_2_bias_v/
+assignvariableop_32_hidden_layer_3_kernel_v-
)assignvariableop_33_hidden_layer_3_bias_v/
+assignvariableop_34_hidden_layer_4_kernel_v-
)assignvariableop_35_hidden_layer_4_bias_v-
)assignvariableop_36_output_layer_kernel_v+
'assignvariableop_37_output_layer_bias_v
identity_39??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:&*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp&assignvariableop_hidden_layer_0_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp&assignvariableop_1_hidden_layer_0_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0?
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0?
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp(assignvariableop_4_hidden_layer_2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp&assignvariableop_5_hidden_layer_2_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0?
AssignVariableOp_6AssignVariableOp(assignvariableop_6_hidden_layer_3_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0?
AssignVariableOp_7AssignVariableOp&assignvariableop_7_hidden_layer_3_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp(assignvariableop_8_hidden_layer_4_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp&assignvariableop_9_hidden_layer_4_biasIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp'assignvariableop_10_output_layer_kernelIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp%assignvariableop_11_output_layer_biasIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0{
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:{
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
_output_shapes
:*
T0?
AssignVariableOp_14AssignVariableOp+assignvariableop_14_hidden_layer_0_kernel_mIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_hidden_layer_0_bias_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
_output_shapes
:*
T0?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_hidden_layer_1_kernel_mIdentity_16:output:0*
_output_shapes
 *
dtype0P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_hidden_layer_1_bias_mIdentity_17:output:0*
_output_shapes
 *
dtype0P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0?
AssignVariableOp_18AssignVariableOp+assignvariableop_18_hidden_layer_2_kernel_mIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_hidden_layer_2_bias_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
_output_shapes
:*
T0?
AssignVariableOp_20AssignVariableOp+assignvariableop_20_hidden_layer_3_kernel_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_hidden_layer_3_bias_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0?
AssignVariableOp_22AssignVariableOp+assignvariableop_22_hidden_layer_4_kernel_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_hidden_layer_4_bias_mIdentity_23:output:0*
_output_shapes
 *
dtype0P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_output_layer_kernel_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_output_layer_bias_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_hidden_layer_0_kernel_vIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_hidden_layer_0_bias_vIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
_output_shapes
:*
T0?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_hidden_layer_1_kernel_vIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_hidden_layer_1_bias_vIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp+assignvariableop_30_hidden_layer_2_kernel_vIdentity_30:output:0*
_output_shapes
 *
dtype0P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_hidden_layer_2_bias_vIdentity_31:output:0*
_output_shapes
 *
dtype0P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_hidden_layer_3_kernel_vIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_hidden_layer_3_bias_vIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
_output_shapes
:*
T0?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_hidden_layer_4_kernel_vIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
_output_shapes
:*
T0?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_hidden_layer_4_bias_vIdentity_35:output:0*
_output_shapes
 *
dtype0P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_output_layer_kernel_vIdentity_36:output:0*
_output_shapes
 *
dtype0P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp'assignvariableop_37_output_layer_bias_vIdentity_37:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:*
dtype0t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueB
B *
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
 ?
Identity_38Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0?
Identity_39IdentityIdentity_38:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_39Identity_39:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372(
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_28: : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :+ '
%
_user_specified_namefile_prefix: 
?	
?
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_6735455

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
??j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?K
?
"__inference__wrapped_model_6734960
input_layer_input<
8sequential_hidden_layer_0_matmul_readvariableop_resource=
9sequential_hidden_layer_0_biasadd_readvariableop_resource<
8sequential_hidden_layer_1_matmul_readvariableop_resource=
9sequential_hidden_layer_1_biasadd_readvariableop_resource<
8sequential_hidden_layer_2_matmul_readvariableop_resource=
9sequential_hidden_layer_2_biasadd_readvariableop_resource<
8sequential_hidden_layer_3_matmul_readvariableop_resource=
9sequential_hidden_layer_3_biasadd_readvariableop_resource<
8sequential_hidden_layer_4_matmul_readvariableop_resource=
9sequential_hidden_layer_4_biasadd_readvariableop_resource:
6sequential_output_layer_matmul_readvariableop_resource;
7sequential_output_layer_biasadd_readvariableop_resource
identity??0sequential/hidden_layer_0/BiasAdd/ReadVariableOp?/sequential/hidden_layer_0/MatMul/ReadVariableOp?0sequential/hidden_layer_1/BiasAdd/ReadVariableOp?/sequential/hidden_layer_1/MatMul/ReadVariableOp?0sequential/hidden_layer_2/BiasAdd/ReadVariableOp?/sequential/hidden_layer_2/MatMul/ReadVariableOp?0sequential/hidden_layer_3/BiasAdd/ReadVariableOp?/sequential/hidden_layer_3/MatMul/ReadVariableOp?0sequential/hidden_layer_4/BiasAdd/ReadVariableOp?/sequential/hidden_layer_4/MatMul/ReadVariableOp?.sequential/output_layer/BiasAdd/ReadVariableOp?-sequential/output_layer/MatMul/ReadVariableOpu
$sequential/input_layer/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"????  ?
sequential/input_layer/ReshapeReshapeinput_layer_input-sequential/input_layer/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0?
/sequential/hidden_layer_0/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
 sequential/hidden_layer_0/MatMulMatMul'sequential/input_layer/Reshape:output:07sequential/hidden_layer_0/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
0sequential/hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
!sequential/hidden_layer_0/BiasAddBiasAdd*sequential/hidden_layer_0/MatMul:product:08sequential/hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential/hidden_layer_0/ReluRelu*sequential/hidden_layer_0/BiasAdd:output:0*(
_output_shapes
:??????????*
T0?
/sequential/hidden_layer_1/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
 sequential/hidden_layer_1/MatMulMatMul,sequential/hidden_layer_0/Relu:activations:07sequential/hidden_layer_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
0sequential/hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
!sequential/hidden_layer_1/BiasAddBiasAdd*sequential/hidden_layer_1/MatMul:product:08sequential/hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential/hidden_layer_1/ReluRelu*sequential/hidden_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
/sequential/hidden_layer_2/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
 sequential/hidden_layer_2/MatMulMatMul,sequential/hidden_layer_1/Relu:activations:07sequential/hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
0sequential/hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
!sequential/hidden_layer_2/BiasAddBiasAdd*sequential/hidden_layer_2/MatMul:product:08sequential/hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential/hidden_layer_2/ReluRelu*sequential/hidden_layer_2/BiasAdd:output:0*(
_output_shapes
:??????????*
T0?
/sequential/hidden_layer_3/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
 sequential/hidden_layer_3/MatMulMatMul,sequential/hidden_layer_2/Relu:activations:07sequential/hidden_layer_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
0sequential/hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
!sequential/hidden_layer_3/BiasAddBiasAdd*sequential/hidden_layer_3/MatMul:product:08sequential/hidden_layer_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential/hidden_layer_3/ReluRelu*sequential/hidden_layer_3/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
/sequential/hidden_layer_4/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
 sequential/hidden_layer_4/MatMulMatMul,sequential/hidden_layer_3/Relu:activations:07sequential/hidden_layer_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
0sequential/hidden_layer_4/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
!sequential/hidden_layer_4/BiasAddBiasAdd*sequential/hidden_layer_4/MatMul:product:08sequential/hidden_layer_4/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
sequential/hidden_layer_4/ReluRelu*sequential/hidden_layer_4/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
-sequential/output_layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0?
sequential/output_layer/MatMulMatMul,sequential/hidden_layer_4/Relu:activations:05sequential/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
.sequential/output_layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
sequential/output_layer/BiasAddBiasAdd(sequential/output_layer/MatMul:product:06sequential/output_layer/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
sequential/output_layer/SoftmaxSoftmax(sequential/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentity)sequential/output_layer/Softmax:softmax:01^sequential/hidden_layer_0/BiasAdd/ReadVariableOp0^sequential/hidden_layer_0/MatMul/ReadVariableOp1^sequential/hidden_layer_1/BiasAdd/ReadVariableOp0^sequential/hidden_layer_1/MatMul/ReadVariableOp1^sequential/hidden_layer_2/BiasAdd/ReadVariableOp0^sequential/hidden_layer_2/MatMul/ReadVariableOp1^sequential/hidden_layer_3/BiasAdd/ReadVariableOp0^sequential/hidden_layer_3/MatMul/ReadVariableOp1^sequential/hidden_layer_4/BiasAdd/ReadVariableOp0^sequential/hidden_layer_4/MatMul/ReadVariableOp/^sequential/output_layer/BiasAdd/ReadVariableOp.^sequential/output_layer/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2`
.sequential/output_layer/BiasAdd/ReadVariableOp.sequential/output_layer/BiasAdd/ReadVariableOp2^
-sequential/output_layer/MatMul/ReadVariableOp-sequential/output_layer/MatMul/ReadVariableOp2d
0sequential/hidden_layer_3/BiasAdd/ReadVariableOp0sequential/hidden_layer_3/BiasAdd/ReadVariableOp2b
/sequential/hidden_layer_2/MatMul/ReadVariableOp/sequential/hidden_layer_2/MatMul/ReadVariableOp2d
0sequential/hidden_layer_1/BiasAdd/ReadVariableOp0sequential/hidden_layer_1/BiasAdd/ReadVariableOp2b
/sequential/hidden_layer_3/MatMul/ReadVariableOp/sequential/hidden_layer_3/MatMul/ReadVariableOp2d
0sequential/hidden_layer_4/BiasAdd/ReadVariableOp0sequential/hidden_layer_4/BiasAdd/ReadVariableOp2b
/sequential/hidden_layer_0/MatMul/ReadVariableOp/sequential/hidden_layer_0/MatMul/ReadVariableOp2d
0sequential/hidden_layer_2/BiasAdd/ReadVariableOp0sequential/hidden_layer_2/BiasAdd/ReadVariableOp2b
/sequential/hidden_layer_4/MatMul/ReadVariableOp/sequential/hidden_layer_4/MatMul/ReadVariableOp2d
0sequential/hidden_layer_0/BiasAdd/ReadVariableOp0sequential/hidden_layer_0/BiasAdd/ReadVariableOp2b
/sequential/hidden_layer_1/MatMul/ReadVariableOp/sequential/hidden_layer_1/MatMul/ReadVariableOp:1 -
+
_user_specified_nameinput_layer_input: : : : : : : : :	 :
 : : 
?
I
-__inference_input_layer_layer_call_fn_6735426

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tout
2*.
_gradient_op_typePartitionedCall-6734978*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_6734972**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:??????????*
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
?	
?
K__inference_hidden_layer_3_layer_call_and_return_conditional_losses_6735080

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*(
_output_shapes
:??????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
I__inference_output_layer_layer_call_and_return_conditional_losses_6735136

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
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
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_6735437

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_6734996

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0Q
ReluReluBiasAdd:output:0*(
_output_shapes
:??????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
d
H__inference_input_layer_layer_call_and_return_conditional_losses_6734972

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
valueB"????  *
dtype0e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:??????????*
T0Y
IdentityIdentityReshape:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?(
?
G__inference_sequential_layer_call_and_return_conditional_losses_6735154
input_layer_input1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_21
-hidden_layer_3_statefulpartitionedcall_args_11
-hidden_layer_3_statefulpartitionedcall_args_21
-hidden_layer_4_statefulpartitionedcall_args_11
-hidden_layer_4_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?&hidden_layer_2/StatefulPartitionedCall?&hidden_layer_3/StatefulPartitionedCall?&hidden_layer_4/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinput_layer_input*
Tin
2*(
_output_shapes
:??????????*
Tout
2*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_6734972**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-6734978?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*(
_output_shapes
:??????????*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_6734996*.
_gradient_op_typePartitionedCall-6735002*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*
Tin
2*.
_gradient_op_typePartitionedCall-6735030*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_6735024*(
_output_shapes
:??????????*
Tout
2**
config_proto

GPU 

CPU2J 8?
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*(
_output_shapes
:??????????*
Tout
2*T
fORM
K__inference_hidden_layer_2_layer_call_and_return_conditional_losses_6735052*
Tin
2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-6735058?
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0-hidden_layer_3_statefulpartitionedcall_args_1-hidden_layer_3_statefulpartitionedcall_args_2*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-6735086*
Tin
2*T
fORM
K__inference_hidden_layer_3_layer_call_and_return_conditional_losses_6735080**
config_proto

GPU 

CPU2J 8*
Tout
2?
&hidden_layer_4/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0-hidden_layer_4_statefulpartitionedcall_args_1-hidden_layer_4_statefulpartitionedcall_args_2*(
_output_shapes
:??????????*T
fORM
K__inference_hidden_layer_4_layer_call_and_return_conditional_losses_6735108*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-6735114?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_4/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6735142*
Tout
2*
Tin
2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_6735136*'
_output_shapes
:?????????
**
config_proto

GPU 

CPU2J 8?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall'^hidden_layer_4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2P
&hidden_layer_4/StatefulPartitionedCall&hidden_layer_4/StatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : : : :	 :
 : : 
?	
?
K__inference_hidden_layer_2_layer_call_and_return_conditional_losses_6735473

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?(
?
G__inference_sequential_layer_call_and_return_conditional_losses_6735179
input_layer_input1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_21
-hidden_layer_3_statefulpartitionedcall_args_11
-hidden_layer_3_statefulpartitionedcall_args_21
-hidden_layer_4_statefulpartitionedcall_args_11
-hidden_layer_4_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?&hidden_layer_2/StatefulPartitionedCall?&hidden_layer_3/StatefulPartitionedCall?&hidden_layer_4/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinput_layer_input*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_6734972*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-6734978*(
_output_shapes
:???????????
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6735002*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_6734996*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*
Tin
2*
Tout
2?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_6735024*
Tout
2*
Tin
2*.
_gradient_op_typePartitionedCall-6735030*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8?
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*(
_output_shapes
:??????????*T
fORM
K__inference_hidden_layer_2_layer_call_and_return_conditional_losses_6735052*.
_gradient_op_typePartitionedCall-6735058*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2?
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0-hidden_layer_3_statefulpartitionedcall_args_1-hidden_layer_3_statefulpartitionedcall_args_2*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*
Tout
2*.
_gradient_op_typePartitionedCall-6735086*
Tin
2*T
fORM
K__inference_hidden_layer_3_layer_call_and_return_conditional_losses_6735080?
&hidden_layer_4/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0-hidden_layer_4_statefulpartitionedcall_args_1-hidden_layer_4_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*.
_gradient_op_typePartitionedCall-6735114*(
_output_shapes
:??????????*
Tout
2*T
fORM
K__inference_hidden_layer_4_layer_call_and_return_conditional_losses_6735108?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_4/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6735142*'
_output_shapes
:?????????
*
Tin
2*
Tout
2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_6735136**
config_proto

GPU 

CPU2J 8?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall'^hidden_layer_4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2P
&hidden_layer_4/StatefulPartitionedCall&hidden_layer_4/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall: :1 -
+
_user_specified_nameinput_layer_input: : : : : : : : :	 :
 : 
?
?
0__inference_hidden_layer_3_layer_call_fn_6735498

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*T
fORM
K__inference_hidden_layer_3_layer_call_and_return_conditional_losses_6735080*
Tin
2*.
_gradient_op_typePartitionedCall-6735086**
config_proto

GPU 

CPU2J 8*
Tout
2*(
_output_shapes
:???????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
,__inference_sequential_layer_call_fn_6735415

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
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_6735248**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
*
Tout
2*.
_gradient_op_typePartitionedCall-6735249?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :	 :
 : : :& "
 
_user_specified_nameinputs: 
?	
?
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_6735024

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*(
_output_shapes
:??????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
%__inference_signature_wrapper_6735283
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12**
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__wrapped_model_6734960*
Tout
2*
Tin
2*.
_gradient_op_typePartitionedCall-6735268*'
_output_shapes
:?????????
?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : : : :	 :
 : : 
?
?
0__inference_hidden_layer_2_layer_call_fn_6735480

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6735058*
Tout
2*
Tin
2*(
_output_shapes
:??????????*T
fORM
K__inference_hidden_layer_2_layer_call_and_return_conditional_losses_6735052**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
,__inference_sequential_layer_call_fn_6735221
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*.
_gradient_op_typePartitionedCall-6735206*'
_output_shapes
:?????????
*
Tout
2**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_6735205*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 :
 : : :1 -
+
_user_specified_nameinput_layer_input: : : 
?	
?
K__inference_hidden_layer_2_layer_call_and_return_conditional_losses_6735052

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
,__inference_sequential_layer_call_fn_6735264
input_layer_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:?????????
*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_6735248*.
_gradient_op_typePartitionedCall-6735249?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:1 -
+
_user_specified_nameinput_layer_input: : : : : : : : :	 :
 : : 
?	
?
K__inference_hidden_layer_4_layer_call_and_return_conditional_losses_6735509

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?	
?
K__inference_hidden_layer_4_layer_call_and_return_conditional_losses_6735108

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
??j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*(
_output_shapes
:??????????*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?'
?
G__inference_sequential_layer_call_and_return_conditional_losses_6735205

inputs1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_21
-hidden_layer_3_statefulpartitionedcall_args_11
-hidden_layer_3_statefulpartitionedcall_args_21
-hidden_layer_4_statefulpartitionedcall_args_11
-hidden_layer_4_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?&hidden_layer_2/StatefulPartitionedCall?&hidden_layer_3/StatefulPartitionedCall?&hidden_layer_4/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinputs*
Tout
2*.
_gradient_op_typePartitionedCall-6734978*
Tin
2*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_6734972*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8?
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_6734996*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-6735002*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_6735024*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-6735030*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2?
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6735058*T
fORM
K__inference_hidden_layer_2_layer_call_and_return_conditional_losses_6735052*
Tin
2*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*
Tout
2?
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0-hidden_layer_3_statefulpartitionedcall_args_1-hidden_layer_3_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tout
2*
Tin
2*T
fORM
K__inference_hidden_layer_3_layer_call_and_return_conditional_losses_6735080*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-6735086?
&hidden_layer_4/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0-hidden_layer_4_statefulpartitionedcall_args_1-hidden_layer_4_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6735114*(
_output_shapes
:??????????*
Tin
2*T
fORM
K__inference_hidden_layer_4_layer_call_and_return_conditional_losses_6735108**
config_proto

GPU 

CPU2J 8*
Tout
2?
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_4/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*
Tin
2*.
_gradient_op_typePartitionedCall-6735142**
config_proto

GPU 

CPU2J 8*
Tout
2*'
_output_shapes
:?????????
*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_6735136?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall'^hidden_layer_4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2P
&hidden_layer_4/StatefulPartitionedCall&hidden_layer_4/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
?	
?
I__inference_output_layer_layer_call_and_return_conditional_losses_6735527

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?
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
SoftmaxSoftmaxBiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
d
H__inference_input_layer_layer_call_and_return_conditional_losses_6735421

inputs
identity^
Reshape/shapeConst*
_output_shapes
:*
valueB"????  *
dtype0e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:??????????*
T0Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0**
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
??
?	
G__inference_sequential_layer_call_and_return_conditional_losses_6735381

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource1
-hidden_layer_3_matmul_readvariableop_resource2
.hidden_layer_3_biasadd_readvariableop_resource1
-hidden_layer_4_matmul_readvariableop_resource2
.hidden_layer_4_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity??%hidden_layer_0/BiasAdd/ReadVariableOp?$hidden_layer_0/MatMul/ReadVariableOp?%hidden_layer_1/BiasAdd/ReadVariableOp?$hidden_layer_1/MatMul/ReadVariableOp?%hidden_layer_2/BiasAdd/ReadVariableOp?$hidden_layer_2/MatMul/ReadVariableOp?%hidden_layer_3/BiasAdd/ReadVariableOp?$hidden_layer_3/MatMul/ReadVariableOp?%hidden_layer_4/BiasAdd/ReadVariableOp?$hidden_layer_4/MatMul/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOpj
input_layer/Reshape/shapeConst*
dtype0*
valueB"????  *
_output_shapes
:}
input_layer/ReshapeReshapeinputs"input_layer/Reshape/shape:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
hidden_layer_0/MatMulMatMulinput_layer/Reshape:output:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*(
_output_shapes
:??????????*
T0?
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
hidden_layer_1/MatMulMatMul!hidden_layer_0/Relu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
hidden_layer_2/MatMulMatMul!hidden_layer_1/Relu:activations:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_3/MatMul/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_3/MatMulMatMul!hidden_layer_2/Relu:activations:0,hidden_layer_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_3/BiasAddBiasAddhidden_layer_3/MatMul:product:0-hidden_layer_3/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_3/ReluReluhidden_layer_3/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_4/MatMul/ReadVariableOpReadVariableOp-hidden_layer_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_4/MatMulMatMul!hidden_layer_3/Relu:activations:0,hidden_layer_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%hidden_layer_4/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
hidden_layer_4/BiasAddBiasAddhidden_layer_4/MatMul:product:0-hidden_layer_4/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_4/ReluReluhidden_layer_4/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0?
output_layer/MatMulMatMul!hidden_layer_4/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
p
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp&^hidden_layer_3/BiasAdd/ReadVariableOp%^hidden_layer_3/MatMul/ReadVariableOp&^hidden_layer_4/BiasAdd/ReadVariableOp%^hidden_layer_4/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp2N
%hidden_layer_3/BiasAdd/ReadVariableOp%hidden_layer_3/BiasAdd/ReadVariableOp2L
$hidden_layer_3/MatMul/ReadVariableOp$hidden_layer_3/MatMul/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2N
%hidden_layer_4/BiasAdd/ReadVariableOp%hidden_layer_4/BiasAdd/ReadVariableOp2L
$hidden_layer_4/MatMul/ReadVariableOp$hidden_layer_4/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
?
?
0__inference_hidden_layer_1_layer_call_fn_6735462

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-6735030*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_6735024*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
0__inference_hidden_layer_0_layer_call_fn_6735444

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*(
_output_shapes
:??????????*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_6734996*
Tout
2*
Tin
2*.
_gradient_op_typePartitionedCall-6735002**
config_proto

GPU 

CPU2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?	
?
K__inference_hidden_layer_3_layer_call_and_return_conditional_losses_6735491

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:?w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
?
.__inference_output_layer_layer_call_fn_6735534

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
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
*.
_gradient_op_typePartitionedCall-6735142*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_6735136?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?
?
0__inference_hidden_layer_4_layer_call_fn_6735516

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6735114*
Tin
2*T
fORM
K__inference_hidden_layer_4_layer_call_and_return_conditional_losses_6735108*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
,__inference_sequential_layer_call_fn_6735398

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
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12**
config_proto

GPU 

CPU2J 8*
Tout
2*P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_6735205*.
_gradient_op_typePartitionedCall-6735206*'
_output_shapes
:?????????
*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
?M
?
 __inference__traced_save_6735673
file_prefix4
0savev2_hidden_layer_0_kernel_read_readvariableop2
.savev2_hidden_layer_0_bias_read_readvariableop4
0savev2_hidden_layer_1_kernel_read_readvariableop2
.savev2_hidden_layer_1_bias_read_readvariableop4
0savev2_hidden_layer_2_kernel_read_readvariableop2
.savev2_hidden_layer_2_bias_read_readvariableop4
0savev2_hidden_layer_3_kernel_read_readvariableop2
.savev2_hidden_layer_3_bias_read_readvariableop4
0savev2_hidden_layer_4_kernel_read_readvariableop2
.savev2_hidden_layer_4_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_hidden_layer_0_kernel_m_read_readvariableop4
0savev2_hidden_layer_0_bias_m_read_readvariableop6
2savev2_hidden_layer_1_kernel_m_read_readvariableop4
0savev2_hidden_layer_1_bias_m_read_readvariableop6
2savev2_hidden_layer_2_kernel_m_read_readvariableop4
0savev2_hidden_layer_2_bias_m_read_readvariableop6
2savev2_hidden_layer_3_kernel_m_read_readvariableop4
0savev2_hidden_layer_3_bias_m_read_readvariableop6
2savev2_hidden_layer_4_kernel_m_read_readvariableop4
0savev2_hidden_layer_4_bias_m_read_readvariableop4
0savev2_output_layer_kernel_m_read_readvariableop2
.savev2_output_layer_bias_m_read_readvariableop6
2savev2_hidden_layer_0_kernel_v_read_readvariableop4
0savev2_hidden_layer_0_bias_v_read_readvariableop6
2savev2_hidden_layer_1_kernel_v_read_readvariableop4
0savev2_hidden_layer_1_bias_v_read_readvariableop6
2savev2_hidden_layer_2_kernel_v_read_readvariableop4
0savev2_hidden_layer_2_bias_v_read_readvariableop6
2savev2_hidden_layer_3_kernel_v_read_readvariableop4
0savev2_hidden_layer_3_bias_v_read_readvariableop6
2savev2_hidden_layer_4_kernel_v_read_readvariableop4
0savev2_hidden_layer_4_bias_v_read_readvariableop4
0savev2_output_layer_kernel_v_read_readvariableop2
.savev2_output_layer_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*<
value3B1 B+_temp_e36a956791f0423490870de6bbb562b1/part*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
value	B :*
_output_shapes
: *
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:&*
dtype0?
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:&*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_hidden_layer_0_kernel_read_readvariableop.savev2_hidden_layer_0_bias_read_readvariableop0savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop0savev2_hidden_layer_2_kernel_read_readvariableop.savev2_hidden_layer_2_bias_read_readvariableop0savev2_hidden_layer_3_kernel_read_readvariableop.savev2_hidden_layer_3_bias_read_readvariableop0savev2_hidden_layer_4_kernel_read_readvariableop.savev2_hidden_layer_4_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_hidden_layer_0_kernel_m_read_readvariableop0savev2_hidden_layer_0_bias_m_read_readvariableop2savev2_hidden_layer_1_kernel_m_read_readvariableop0savev2_hidden_layer_1_bias_m_read_readvariableop2savev2_hidden_layer_2_kernel_m_read_readvariableop0savev2_hidden_layer_2_bias_m_read_readvariableop2savev2_hidden_layer_3_kernel_m_read_readvariableop0savev2_hidden_layer_3_bias_m_read_readvariableop2savev2_hidden_layer_4_kernel_m_read_readvariableop0savev2_hidden_layer_4_bias_m_read_readvariableop0savev2_output_layer_kernel_m_read_readvariableop.savev2_output_layer_bias_m_read_readvariableop2savev2_hidden_layer_0_kernel_v_read_readvariableop0savev2_hidden_layer_0_bias_v_read_readvariableop2savev2_hidden_layer_1_kernel_v_read_readvariableop0savev2_hidden_layer_1_bias_v_read_readvariableop2savev2_hidden_layer_2_kernel_v_read_readvariableop0savev2_hidden_layer_2_bias_v_read_readvariableop2savev2_hidden_layer_3_kernel_v_read_readvariableop0savev2_hidden_layer_3_bias_v_read_readvariableop2savev2_hidden_layer_4_kernel_v_read_readvariableop0savev2_hidden_layer_4_bias_v_read_readvariableop0savev2_output_layer_kernel_v_read_readvariableop.savev2_output_layer_bias_v_read_readvariableop"/device:CPU:0*4
dtypes*
(2&*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B :?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:q
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:?:
??:?:
??:?:
??:?:
??:?:	?
:
: : :
??:?:
??:?:
??:?:
??:?:
??:?:	?
:
:
??:?:
??:?:
??:?:
??:?:
??:?:	?
:
: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' 
??
?	
G__inference_sequential_layer_call_and_return_conditional_losses_6735333

inputs1
-hidden_layer_0_matmul_readvariableop_resource2
.hidden_layer_0_biasadd_readvariableop_resource1
-hidden_layer_1_matmul_readvariableop_resource2
.hidden_layer_1_biasadd_readvariableop_resource1
-hidden_layer_2_matmul_readvariableop_resource2
.hidden_layer_2_biasadd_readvariableop_resource1
-hidden_layer_3_matmul_readvariableop_resource2
.hidden_layer_3_biasadd_readvariableop_resource1
-hidden_layer_4_matmul_readvariableop_resource2
.hidden_layer_4_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identity??%hidden_layer_0/BiasAdd/ReadVariableOp?$hidden_layer_0/MatMul/ReadVariableOp?%hidden_layer_1/BiasAdd/ReadVariableOp?$hidden_layer_1/MatMul/ReadVariableOp?%hidden_layer_2/BiasAdd/ReadVariableOp?$hidden_layer_2/MatMul/ReadVariableOp?%hidden_layer_3/BiasAdd/ReadVariableOp?$hidden_layer_3/MatMul/ReadVariableOp?%hidden_layer_4/BiasAdd/ReadVariableOp?$hidden_layer_4/MatMul/ReadVariableOp?#output_layer/BiasAdd/ReadVariableOp?"output_layer/MatMul/ReadVariableOpj
input_layer/Reshape/shapeConst*
valueB"????  *
dtype0*
_output_shapes
:}
input_layer/ReshapeReshapeinputs"input_layer/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0?
$hidden_layer_0/MatMul/ReadVariableOpReadVariableOp-hidden_layer_0_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_0/MatMulMatMulinput_layer/Reshape:output:0,hidden_layer_0/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
%hidden_layer_0/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_0_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
hidden_layer_0/BiasAddBiasAddhidden_layer_0/MatMul:product:0-hidden_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
hidden_layer_0/ReluReluhidden_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_1/MatMulMatMul!hidden_layer_0/Relu:activations:0,hidden_layer_1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
%hidden_layer_1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_1/BiasAddBiasAddhidden_layer_1/MatMul:product:0-hidden_layer_1/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_1/ReluReluhidden_layer_1/BiasAdd:output:0*(
_output_shapes
:??????????*
T0?
$hidden_layer_2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
hidden_layer_2/MatMulMatMul!hidden_layer_1/Relu:activations:0,hidden_layer_2/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
%hidden_layer_2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
hidden_layer_2/BiasAddBiasAddhidden_layer_2/MatMul:product:0-hidden_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
hidden_layer_2/ReluReluhidden_layer_2/BiasAdd:output:0*(
_output_shapes
:??????????*
T0?
$hidden_layer_3/MatMul/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
hidden_layer_3/MatMulMatMul!hidden_layer_2/Relu:activations:0,hidden_layer_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%hidden_layer_3/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:?*
dtype0?
hidden_layer_3/BiasAddBiasAddhidden_layer_3/MatMul:product:0-hidden_layer_3/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_3/ReluReluhidden_layer_3/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
$hidden_layer_4/MatMul/ReadVariableOpReadVariableOp-hidden_layer_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
??*
dtype0?
hidden_layer_4/MatMulMatMul!hidden_layer_3/Relu:activations:0,hidden_layer_4/MatMul/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0?
%hidden_layer_4/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
hidden_layer_4/BiasAddBiasAddhidden_layer_4/MatMul:product:0-hidden_layer_4/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:??????????*
T0o
hidden_layer_4/ReluReluhidden_layer_4/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0?
output_layer/MatMulMatMul!hidden_layer_4/Relu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
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
T0?
IdentityIdentityoutput_layer/Softmax:softmax:0&^hidden_layer_0/BiasAdd/ReadVariableOp%^hidden_layer_0/MatMul/ReadVariableOp&^hidden_layer_1/BiasAdd/ReadVariableOp%^hidden_layer_1/MatMul/ReadVariableOp&^hidden_layer_2/BiasAdd/ReadVariableOp%^hidden_layer_2/MatMul/ReadVariableOp&^hidden_layer_3/BiasAdd/ReadVariableOp%^hidden_layer_3/MatMul/ReadVariableOp&^hidden_layer_4/BiasAdd/ReadVariableOp%^hidden_layer_4/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2N
%hidden_layer_0/BiasAdd/ReadVariableOp%hidden_layer_0/BiasAdd/ReadVariableOp2L
$hidden_layer_2/MatMul/ReadVariableOp$hidden_layer_2/MatMul/ReadVariableOp2N
%hidden_layer_3/BiasAdd/ReadVariableOp%hidden_layer_3/BiasAdd/ReadVariableOp2L
$hidden_layer_3/MatMul/ReadVariableOp$hidden_layer_3/MatMul/ReadVariableOp2N
%hidden_layer_1/BiasAdd/ReadVariableOp%hidden_layer_1/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2L
$hidden_layer_0/MatMul/ReadVariableOp$hidden_layer_0/MatMul/ReadVariableOp2N
%hidden_layer_4/BiasAdd/ReadVariableOp%hidden_layer_4/BiasAdd/ReadVariableOp2L
$hidden_layer_4/MatMul/ReadVariableOp$hidden_layer_4/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2L
$hidden_layer_1/MatMul/ReadVariableOp$hidden_layer_1/MatMul/ReadVariableOp2N
%hidden_layer_2/BiasAdd/ReadVariableOp%hidden_layer_2/BiasAdd/ReadVariableOp:
 : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 
?'
?
G__inference_sequential_layer_call_and_return_conditional_losses_6735248

inputs1
-hidden_layer_0_statefulpartitionedcall_args_11
-hidden_layer_0_statefulpartitionedcall_args_21
-hidden_layer_1_statefulpartitionedcall_args_11
-hidden_layer_1_statefulpartitionedcall_args_21
-hidden_layer_2_statefulpartitionedcall_args_11
-hidden_layer_2_statefulpartitionedcall_args_21
-hidden_layer_3_statefulpartitionedcall_args_11
-hidden_layer_3_statefulpartitionedcall_args_21
-hidden_layer_4_statefulpartitionedcall_args_11
-hidden_layer_4_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identity??&hidden_layer_0/StatefulPartitionedCall?&hidden_layer_1/StatefulPartitionedCall?&hidden_layer_2/StatefulPartitionedCall?&hidden_layer_3/StatefulPartitionedCall?&hidden_layer_4/StatefulPartitionedCall?$output_layer/StatefulPartitionedCall?
input_layer/PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-6734978**
config_proto

GPU 

CPU2J 8*
Tin
2*Q
fLRJ
H__inference_input_layer_layer_call_and_return_conditional_losses_6734972*
Tout
2*(
_output_shapes
:???????????
&hidden_layer_0/StatefulPartitionedCallStatefulPartitionedCall$input_layer/PartitionedCall:output:0-hidden_layer_0_statefulpartitionedcall_args_1-hidden_layer_0_statefulpartitionedcall_args_2*
Tout
2*T
fORM
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_6734996*(
_output_shapes
:??????????**
config_proto

GPU 

CPU2J 8*
Tin
2*.
_gradient_op_typePartitionedCall-6735002?
&hidden_layer_1/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_0/StatefulPartitionedCall:output:0-hidden_layer_1_statefulpartitionedcall_args_1-hidden_layer_1_statefulpartitionedcall_args_2*
Tin
2*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-6735030*T
fORM
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_6735024*
Tout
2**
config_proto

GPU 

CPU2J 8?
&hidden_layer_2/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_1/StatefulPartitionedCall:output:0-hidden_layer_2_statefulpartitionedcall_args_1-hidden_layer_2_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*
Tout
2*T
fORM
K__inference_hidden_layer_2_layer_call_and_return_conditional_losses_6735052*(
_output_shapes
:??????????*.
_gradient_op_typePartitionedCall-6735058?
&hidden_layer_3/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_2/StatefulPartitionedCall:output:0-hidden_layer_3_statefulpartitionedcall_args_1-hidden_layer_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*T
fORM
K__inference_hidden_layer_3_layer_call_and_return_conditional_losses_6735080**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-6735086*(
_output_shapes
:???????????
&hidden_layer_4/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_3/StatefulPartitionedCall:output:0-hidden_layer_4_statefulpartitionedcall_args_1-hidden_layer_4_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_hidden_layer_4_layer_call_and_return_conditional_losses_6735108*.
_gradient_op_typePartitionedCall-6735114*
Tin
2*
Tout
2*(
_output_shapes
:???????????
$output_layer/StatefulPartitionedCallStatefulPartitionedCall/hidden_layer_4/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*R
fMRK
I__inference_output_layer_layer_call_and_return_conditional_losses_6735136**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:?????????
*.
_gradient_op_typePartitionedCall-6735142?
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0'^hidden_layer_0/StatefulPartitionedCall'^hidden_layer_1/StatefulPartitionedCall'^hidden_layer_2/StatefulPartitionedCall'^hidden_layer_3/StatefulPartitionedCall'^hidden_layer_4/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*Z
_input_shapesI
G:?????????::::::::::::2P
&hidden_layer_0/StatefulPartitionedCall&hidden_layer_0/StatefulPartitionedCall2P
&hidden_layer_1/StatefulPartitionedCall&hidden_layer_1/StatefulPartitionedCall2P
&hidden_layer_2/StatefulPartitionedCall&hidden_layer_2/StatefulPartitionedCall2P
&hidden_layer_3/StatefulPartitionedCall&hidden_layer_3/StatefulPartitionedCall2P
&hidden_layer_4/StatefulPartitionedCall&hidden_layer_4/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall: : : : : : : :	 :
 : : :& "
 
_user_specified_nameinputs: "wL
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
?5
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		optimizer

	variables
regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?1
_tf_keras_sequential?1{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Flatten", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_3", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_4", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Flatten", "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_3", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_layer_4", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "input_layer_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28, 28], "config": {"batch_input_shape": [null, 28, 28], "dtype": "float32", "sparse": false, "name": "input_layer_input"}}
?
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "input_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 28, 28], "config": {"name": "input_layer", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_0", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}}
?

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_1", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 304}}}}
?

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_2", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 304}}}}
?

)kernel
*bias
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_3", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 304}}}}
?

/kernel
0bias
1	variables
2regularization_losses
3trainable_variables
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "hidden_layer_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "hidden_layer_4", "trainable": true, "dtype": "float32", "units": 304, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 304}}}}
?

5kernel
6bias
7	variables
8regularization_losses
9trainable_variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 304}}}}
?mkmlmmmn#mo$mp)mq*mr/ms0mt5mu6mvvwvxvyvz#v{$v|)v}*v~/v0v?5v?6v?"
	optimizer
v
0
1
2
3
#4
$5
)6
*7
/8
09
510
611"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
#4
$5
)6
*7
/8
09
510
611"
trackable_list_wrapper
?

	variables
regularization_losses

;layers
trainable_variables
<non_trainable_variables
=layer_regularization_losses
>metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
regularization_losses

?layers
trainable_variables
@metrics
Alayer_regularization_losses
Bnon_trainable_variables
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
?
	variables
regularization_losses

Clayers
trainable_variables
Dmetrics
Elayer_regularization_losses
Fnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
??2hidden_layer_0/kernel
": ?2hidden_layer_0/bias
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
	variables
regularization_losses

Glayers
trainable_variables
Hmetrics
Ilayer_regularization_losses
Jnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
??2hidden_layer_1/kernel
": ?2hidden_layer_1/bias
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
	variables
 regularization_losses

Klayers
!trainable_variables
Lmetrics
Mlayer_regularization_losses
Nnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
??2hidden_layer_2/kernel
": ?2hidden_layer_2/bias
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
%	variables
&regularization_losses

Olayers
'trainable_variables
Pmetrics
Qlayer_regularization_losses
Rnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
??2hidden_layer_3/kernel
": ?2hidden_layer_3/bias
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
+	variables
,regularization_losses

Slayers
-trainable_variables
Tmetrics
Ulayer_regularization_losses
Vnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'
??2hidden_layer_4/kernel
": ?2hidden_layer_4/bias
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
1	variables
2regularization_losses

Wlayers
3trainable_variables
Xmetrics
Ylayer_regularization_losses
Znon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$	?
2output_layer/kernel
:
2output_layer/bias
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
7	variables
8regularization_losses

[layers
9trainable_variables
\metrics
]layer_regularization_losses
^non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
_0"
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
	`total
	acount
b
_fn_kwargs
c	variables
dregularization_losses
etrainable_variables
f	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
c	variables
dregularization_losses

glayers
etrainable_variables
hmetrics
ilayer_regularization_losses
jnon_trainable_variables
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
`0
a1"
trackable_list_wrapper
):'
??2hidden_layer_0/kernel/m
": ?2hidden_layer_0/bias/m
):'
??2hidden_layer_1/kernel/m
": ?2hidden_layer_1/bias/m
):'
??2hidden_layer_2/kernel/m
": ?2hidden_layer_2/bias/m
):'
??2hidden_layer_3/kernel/m
": ?2hidden_layer_3/bias/m
):'
??2hidden_layer_4/kernel/m
": ?2hidden_layer_4/bias/m
&:$	?
2output_layer/kernel/m
:
2output_layer/bias/m
):'
??2hidden_layer_0/kernel/v
": ?2hidden_layer_0/bias/v
):'
??2hidden_layer_1/kernel/v
": ?2hidden_layer_1/bias/v
):'
??2hidden_layer_2/kernel/v
": ?2hidden_layer_2/bias/v
):'
??2hidden_layer_3/kernel/v
": ?2hidden_layer_3/bias/v
):'
??2hidden_layer_4/kernel/v
": ?2hidden_layer_4/bias/v
&:$	?
2output_layer/kernel/v
:
2output_layer/bias/v
?2?
"__inference__wrapped_model_6734960?
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
?2?
G__inference_sequential_layer_call_and_return_conditional_losses_6735381
G__inference_sequential_layer_call_and_return_conditional_losses_6735333
G__inference_sequential_layer_call_and_return_conditional_losses_6735179
G__inference_sequential_layer_call_and_return_conditional_losses_6735154?
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
?2?
,__inference_sequential_layer_call_fn_6735415
,__inference_sequential_layer_call_fn_6735264
,__inference_sequential_layer_call_fn_6735221
,__inference_sequential_layer_call_fn_6735398?
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
H__inference_input_layer_layer_call_and_return_conditional_losses_6735421?
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
-__inference_input_layer_layer_call_fn_6735426?
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
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_6735437?
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
0__inference_hidden_layer_0_layer_call_fn_6735444?
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
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_6735455?
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
0__inference_hidden_layer_1_layer_call_fn_6735462?
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
K__inference_hidden_layer_2_layer_call_and_return_conditional_losses_6735473?
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
0__inference_hidden_layer_2_layer_call_fn_6735480?
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
K__inference_hidden_layer_3_layer_call_and_return_conditional_losses_6735491?
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
0__inference_hidden_layer_3_layer_call_fn_6735498?
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
K__inference_hidden_layer_4_layer_call_and_return_conditional_losses_6735509?
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
0__inference_hidden_layer_4_layer_call_fn_6735516?
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
I__inference_output_layer_layer_call_and_return_conditional_losses_6735527?
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
.__inference_output_layer_layer_call_fn_6735534?
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
%__inference_signature_wrapper_6735283input_layer_input
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
G__inference_sequential_layer_call_and_return_conditional_losses_6735333r#$)*/056;?8
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
,__inference_sequential_layer_call_fn_6735221p#$)*/056F?C
<?9
/?,
input_layer_input?????????
p

 
? "??????????
?
G__inference_sequential_layer_call_and_return_conditional_losses_6735179}#$)*/056F?C
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
? ?
.__inference_output_layer_layer_call_fn_6735534P560?-
&?#
!?
inputs??????????
? "??????????
?
G__inference_sequential_layer_call_and_return_conditional_losses_6735154}#$)*/056F?C
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
K__inference_hidden_layer_0_layer_call_and_return_conditional_losses_6735437^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
0__inference_hidden_layer_2_layer_call_fn_6735480Q#$0?-
&?#
!?
inputs??????????
? "????????????
H__inference_input_layer_layer_call_and_return_conditional_losses_6735421]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? ?
G__inference_sequential_layer_call_and_return_conditional_losses_6735381r#$)*/056;?8
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
0__inference_hidden_layer_4_layer_call_fn_6735516Q/00?-
&?#
!?
inputs??????????
? "????????????
-__inference_input_layer_layer_call_fn_6735426P3?0
)?&
$?!
inputs?????????
? "????????????
I__inference_output_layer_layer_call_and_return_conditional_losses_6735527]560?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? ?
K__inference_hidden_layer_2_layer_call_and_return_conditional_losses_6735473^#$0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
0__inference_hidden_layer_3_layer_call_fn_6735498Q)*0?-
&?#
!?
inputs??????????
? "????????????
"__inference__wrapped_model_6734960?#$)*/056>?;
4?1
/?,
input_layer_input?????????
? ";?8
6
output_layer&?#
output_layer?????????
?
0__inference_hidden_layer_0_layer_call_fn_6735444Q0?-
&?#
!?
inputs??????????
? "????????????
,__inference_sequential_layer_call_fn_6735264p#$)*/056F?C
<?9
/?,
input_layer_input?????????
p 

 
? "??????????
?
K__inference_hidden_layer_4_layer_call_and_return_conditional_losses_6735509^/00?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_sequential_layer_call_fn_6735415e#$)*/056;?8
1?.
$?!
inputs?????????
p 

 
? "??????????
?
,__inference_sequential_layer_call_fn_6735398e#$)*/056;?8
1?.
$?!
inputs?????????
p

 
? "??????????
?
K__inference_hidden_layer_1_layer_call_and_return_conditional_losses_6735455^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
K__inference_hidden_layer_3_layer_call_and_return_conditional_losses_6735491^)*0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
%__inference_signature_wrapper_6735283?#$)*/056S?P
? 
I?F
D
input_layer_input/?,
input_layer_input?????????";?8
6
output_layer&?#
output_layer?????????
?
0__inference_hidden_layer_1_layer_call_fn_6735462Q0?-
&?#
!?
inputs??????????
? "???????????