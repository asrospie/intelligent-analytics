½É
Ô
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68¨¿
v
enc_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameenc_1/kernel
o
 enc_1/kernel/Read/ReadVariableOpReadVariableOpenc_1/kernel* 
_output_shapes
:
*
dtype0
m

enc_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
enc_1/bias
f
enc_1/bias/Read/ReadVariableOpReadVariableOp
enc_1/bias*
_output_shapes	
:*
dtype0
v
enc_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*
shared_nameenc_2/kernel
o
 enc_2/kernel/Read/ReadVariableOpReadVariableOpenc_2/kernel* 
_output_shapes
:
Ä*
dtype0
m

enc_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*
shared_name
enc_2/bias
f
enc_2/bias/Read/ReadVariableOpReadVariableOp
enc_2/bias*
_output_shapes	
:Ä*
dtype0
u
enc_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Äb*
shared_nameenc_3/kernel
n
 enc_3/kernel/Read/ReadVariableOpReadVariableOpenc_3/kernel*
_output_shapes
:	Äb*
dtype0
l

enc_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:b*
shared_name
enc_3/bias
e
enc_3/bias/Read/ReadVariableOpReadVariableOp
enc_3/bias*
_output_shapes
:b*
dtype0
u
dec_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	bÄ*
shared_namedec_3/kernel
n
 dec_3/kernel/Read/ReadVariableOpReadVariableOpdec_3/kernel*
_output_shapes
:	bÄ*
dtype0
m

dec_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*
shared_name
dec_3/bias
f
dec_3/bias/Read/ReadVariableOpReadVariableOp
dec_3/bias*
_output_shapes	
:Ä*
dtype0
v
dec_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*
shared_namedec_2/kernel
o
 dec_2/kernel/Read/ReadVariableOpReadVariableOpdec_2/kernel* 
_output_shapes
:
Ä*
dtype0
m

dec_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dec_2/bias
f
dec_2/bias/Read/ReadVariableOpReadVariableOp
dec_2/bias*
_output_shapes	
:*
dtype0

decoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_namedecoder_output/kernel

)decoder_output/kernel/Read/ReadVariableOpReadVariableOpdecoder_output/kernel* 
_output_shapes
:
*
dtype0

decoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namedecoder_output/bias
x
'decoder_output/bias/Read/ReadVariableOpReadVariableOpdecoder_output/bias*
_output_shapes	
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/enc_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/enc_1/kernel/m
}
'Adam/enc_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_1/kernel/m* 
_output_shapes
:
*
dtype0
{
Adam/enc_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/enc_1/bias/m
t
%Adam/enc_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_1/bias/m*
_output_shapes	
:*
dtype0

Adam/enc_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*$
shared_nameAdam/enc_2/kernel/m
}
'Adam/enc_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_2/kernel/m* 
_output_shapes
:
Ä*
dtype0
{
Adam/enc_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*"
shared_nameAdam/enc_2/bias/m
t
%Adam/enc_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_2/bias/m*
_output_shapes	
:Ä*
dtype0

Adam/enc_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Äb*$
shared_nameAdam/enc_3/kernel/m
|
'Adam/enc_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/enc_3/kernel/m*
_output_shapes
:	Äb*
dtype0
z
Adam/enc_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:b*"
shared_nameAdam/enc_3/bias/m
s
%Adam/enc_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/enc_3/bias/m*
_output_shapes
:b*
dtype0

Adam/dec_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	bÄ*$
shared_nameAdam/dec_3/kernel/m
|
'Adam/dec_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_3/kernel/m*
_output_shapes
:	bÄ*
dtype0
{
Adam/dec_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*"
shared_nameAdam/dec_3/bias/m
t
%Adam/dec_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_3/bias/m*
_output_shapes	
:Ä*
dtype0

Adam/dec_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*$
shared_nameAdam/dec_2/kernel/m
}
'Adam/dec_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dec_2/kernel/m* 
_output_shapes
:
Ä*
dtype0
{
Adam/dec_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dec_2/bias/m
t
%Adam/dec_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dec_2/bias/m*
_output_shapes	
:*
dtype0

Adam/decoder_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/decoder_output/kernel/m

0Adam/decoder_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoder_output/kernel/m* 
_output_shapes
:
*
dtype0

Adam/decoder_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/decoder_output/bias/m

.Adam/decoder_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoder_output/bias/m*
_output_shapes	
:*
dtype0

Adam/enc_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/enc_1/kernel/v
}
'Adam/enc_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_1/kernel/v* 
_output_shapes
:
*
dtype0
{
Adam/enc_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/enc_1/bias/v
t
%Adam/enc_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_1/bias/v*
_output_shapes	
:*
dtype0

Adam/enc_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*$
shared_nameAdam/enc_2/kernel/v
}
'Adam/enc_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_2/kernel/v* 
_output_shapes
:
Ä*
dtype0
{
Adam/enc_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*"
shared_nameAdam/enc_2/bias/v
t
%Adam/enc_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_2/bias/v*
_output_shapes	
:Ä*
dtype0

Adam/enc_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Äb*$
shared_nameAdam/enc_3/kernel/v
|
'Adam/enc_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/enc_3/kernel/v*
_output_shapes
:	Äb*
dtype0
z
Adam/enc_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:b*"
shared_nameAdam/enc_3/bias/v
s
%Adam/enc_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/enc_3/bias/v*
_output_shapes
:b*
dtype0

Adam/dec_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	bÄ*$
shared_nameAdam/dec_3/kernel/v
|
'Adam/dec_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_3/kernel/v*
_output_shapes
:	bÄ*
dtype0
{
Adam/dec_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*"
shared_nameAdam/dec_3/bias/v
t
%Adam/dec_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_3/bias/v*
_output_shapes	
:Ä*
dtype0

Adam/dec_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ä*$
shared_nameAdam/dec_2/kernel/v
}
'Adam/dec_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dec_2/kernel/v* 
_output_shapes
:
Ä*
dtype0
{
Adam/dec_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dec_2/bias/v
t
%Adam/dec_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dec_2/bias/v*
_output_shapes	
:*
dtype0

Adam/decoder_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/decoder_output/kernel/v

0Adam/decoder_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoder_output/kernel/v* 
_output_shapes
:
*
dtype0

Adam/decoder_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/decoder_output/bias/v

.Adam/decoder_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoder_output/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
þI
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¹I
value¯IB¬I B¥I
Ã
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
layer_with_weights-5
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*
¦

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses*
¦

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
¦

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses*
¦

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses*
£
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratemompmqmr!ms"mt)mu*mv1mw2mx9my:mzv{v|v}v~!v"v)v*v1v2v9v:v*
Z
0
1
2
3
!4
"5
)6
*7
18
29
910
:11*
Z
0
1
2
3
!4
"5
)6
*7
18
29
910
:11*
* 
°
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Kserving_default* 
\V
VARIABLE_VALUEenc_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEenc_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEenc_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
enc_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

!0
"1*

!0
"1*
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEdec_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dec_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 

[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
\V
VARIABLE_VALUEdec_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dec_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

10
21*

10
21*
* 

`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEdecoder_output/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEdecoder_output/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

90
:1*

90
:1*
* 

enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*

j0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	ktotal
	lcount
m	variables
n	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

k0
l1*

m	variables*
y
VARIABLE_VALUEAdam/enc_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dec_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dec_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dec_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dec_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/decoder_output/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/decoder_output/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/enc_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/enc_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dec_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dec_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dec_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dec_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/decoder_output/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/decoder_output/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_layerPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ý
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biasenc_3/kernel
enc_3/biasdec_3/kernel
dec_3/biasdec_2/kernel
dec_2/biasdecoder_output/kerneldecoder_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_110211
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
§
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename enc_1/kernel/Read/ReadVariableOpenc_1/bias/Read/ReadVariableOp enc_2/kernel/Read/ReadVariableOpenc_2/bias/Read/ReadVariableOp enc_3/kernel/Read/ReadVariableOpenc_3/bias/Read/ReadVariableOp dec_3/kernel/Read/ReadVariableOpdec_3/bias/Read/ReadVariableOp dec_2/kernel/Read/ReadVariableOpdec_2/bias/Read/ReadVariableOp)decoder_output/kernel/Read/ReadVariableOp'decoder_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/enc_1/kernel/m/Read/ReadVariableOp%Adam/enc_1/bias/m/Read/ReadVariableOp'Adam/enc_2/kernel/m/Read/ReadVariableOp%Adam/enc_2/bias/m/Read/ReadVariableOp'Adam/enc_3/kernel/m/Read/ReadVariableOp%Adam/enc_3/bias/m/Read/ReadVariableOp'Adam/dec_3/kernel/m/Read/ReadVariableOp%Adam/dec_3/bias/m/Read/ReadVariableOp'Adam/dec_2/kernel/m/Read/ReadVariableOp%Adam/dec_2/bias/m/Read/ReadVariableOp0Adam/decoder_output/kernel/m/Read/ReadVariableOp.Adam/decoder_output/bias/m/Read/ReadVariableOp'Adam/enc_1/kernel/v/Read/ReadVariableOp%Adam/enc_1/bias/v/Read/ReadVariableOp'Adam/enc_2/kernel/v/Read/ReadVariableOp%Adam/enc_2/bias/v/Read/ReadVariableOp'Adam/enc_3/kernel/v/Read/ReadVariableOp%Adam/enc_3/bias/v/Read/ReadVariableOp'Adam/dec_3/kernel/v/Read/ReadVariableOp%Adam/dec_3/bias/v/Read/ReadVariableOp'Adam/dec_2/kernel/v/Read/ReadVariableOp%Adam/dec_2/bias/v/Read/ReadVariableOp0Adam/decoder_output/kernel/v/Read/ReadVariableOp.Adam/decoder_output/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_110483
Æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biasenc_3/kernel
enc_3/biasdec_3/kernel
dec_3/biasdec_2/kernel
dec_2/biasdecoder_output/kerneldecoder_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/enc_1/kernel/mAdam/enc_1/bias/mAdam/enc_2/kernel/mAdam/enc_2/bias/mAdam/enc_3/kernel/mAdam/enc_3/bias/mAdam/dec_3/kernel/mAdam/dec_3/bias/mAdam/dec_2/kernel/mAdam/dec_2/bias/mAdam/decoder_output/kernel/mAdam/decoder_output/bias/mAdam/enc_1/kernel/vAdam/enc_1/bias/vAdam/enc_2/kernel/vAdam/enc_2/bias/vAdam/enc_3/kernel/vAdam/enc_3/bias/vAdam/dec_3/kernel/vAdam/dec_3/bias/vAdam/dec_2/kernel/vAdam/dec_2/bias/vAdam/decoder_output/kernel/vAdam/decoder_output/bias/v*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_110622¶

¸
(__inference_model_2_layer_call_fn_109775
input_layer
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:	Äb
	unknown_4:b
	unknown_5:	bÄ
	unknown_6:	Ä
	unknown_7:
Ä
	unknown_8:	
	unknown_9:


unknown_10:	
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_109748p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
Õ

/__inference_decoder_output_layer_call_fn_110320

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_109741p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã

&__inference_enc_1_layer_call_fn_110220

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_1_layer_call_and_return_conditional_losses_109656p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

õ
A__inference_enc_1_layer_call_and_return_conditional_losses_110231

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
¹
C__inference_model_2_layer_call_and_return_conditional_losses_110024
input_layer 
enc_1_109993:

enc_1_109995:	 
enc_2_109998:
Ä
enc_2_110000:	Ä
enc_3_110003:	Äb
enc_3_110005:b
dec_3_110008:	bÄ
dec_3_110010:	Ä 
dec_2_110013:
Ä
dec_2_110015:	)
decoder_output_110018:
$
decoder_output_110020:	
identity¢dec_2/StatefulPartitionedCall¢dec_3/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢enc_3/StatefulPartitionedCallê
enc_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerenc_1_109993enc_1_109995*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_1_layer_call_and_return_conditional_losses_109656
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_109998enc_2_110000*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_2_layer_call_and_return_conditional_losses_109673
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_110003enc_3_110005*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_3_layer_call_and_return_conditional_losses_109690
dec_3/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0dec_3_110008dec_3_110010*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_3_layer_call_and_return_conditional_losses_109707
dec_2/StatefulPartitionedCallStatefulPartitionedCall&dec_3/StatefulPartitionedCall:output:0dec_2_110013dec_2_110015*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_2_layer_call_and_return_conditional_losses_109724©
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall&dec_2/StatefulPartitionedCall:output:0decoder_output_110018decoder_output_110020*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_109741
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dec_2/StatefulPartitionedCall^dec_3/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dec_2/StatefulPartitionedCalldec_2/StatefulPartitionedCall2>
dec_3/StatefulPartitionedCalldec_3/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer


ó
A__inference_enc_3_layer_call_and_return_conditional_losses_109690

inputs1
matmul_readvariableop_resource:	Äb-
biasadd_readvariableop_resource:b
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Äb*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:b*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿba
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
¿

&__inference_enc_3_layer_call_fn_110260

inputs
unknown:	Äb
	unknown_0:b
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_3_layer_call_and_return_conditional_losses_109690o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
¤

õ
A__inference_dec_2_layer_call_and_return_conditional_losses_110311

inputs2
matmul_readvariableop_resource:
Ä.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
¤

õ
A__inference_dec_2_layer_call_and_return_conditional_losses_109724

inputs2
matmul_readvariableop_resource:
Ä.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
À

&__inference_dec_3_layer_call_fn_110280

inputs
unknown:	bÄ
	unknown_0:	Ä
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_3_layer_call_and_return_conditional_losses_109707p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
¤

õ
A__inference_enc_2_layer_call_and_return_conditional_losses_110251

inputs2
matmul_readvariableop_resource:
Ä.
biasadd_readvariableop_resource:	Ä
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â

´
$__inference_signature_wrapper_110211
input_layer
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:	Äb
	unknown_4:b
	unknown_5:	bÄ
	unknown_6:	Ä
	unknown_7:
Ä
	unknown_8:	
	unknown_9:


unknown_10:	
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_109638p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
 

ô
A__inference_dec_3_layer_call_and_return_conditional_losses_110291

inputs1
matmul_readvariableop_resource:	bÄ.
biasadd_readvariableop_resource:	Ä
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	bÄ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
Ã

&__inference_dec_2_layer_call_fn_110300

inputs
unknown:
Ä
	unknown_0:	
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_2_layer_call_and_return_conditional_losses_109724p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
¾«
÷
"__inference__traced_restore_110622
file_prefix1
assignvariableop_enc_1_kernel:
,
assignvariableop_1_enc_1_bias:	3
assignvariableop_2_enc_2_kernel:
Ä,
assignvariableop_3_enc_2_bias:	Ä2
assignvariableop_4_enc_3_kernel:	Äb+
assignvariableop_5_enc_3_bias:b2
assignvariableop_6_dec_3_kernel:	bÄ,
assignvariableop_7_dec_3_bias:	Ä3
assignvariableop_8_dec_2_kernel:
Ä,
assignvariableop_9_dec_2_bias:	=
)assignvariableop_10_decoder_output_kernel:
6
'assignvariableop_11_decoder_output_bias:	'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: ;
'assignvariableop_19_adam_enc_1_kernel_m:
4
%assignvariableop_20_adam_enc_1_bias_m:	;
'assignvariableop_21_adam_enc_2_kernel_m:
Ä4
%assignvariableop_22_adam_enc_2_bias_m:	Ä:
'assignvariableop_23_adam_enc_3_kernel_m:	Äb3
%assignvariableop_24_adam_enc_3_bias_m:b:
'assignvariableop_25_adam_dec_3_kernel_m:	bÄ4
%assignvariableop_26_adam_dec_3_bias_m:	Ä;
'assignvariableop_27_adam_dec_2_kernel_m:
Ä4
%assignvariableop_28_adam_dec_2_bias_m:	D
0assignvariableop_29_adam_decoder_output_kernel_m:
=
.assignvariableop_30_adam_decoder_output_bias_m:	;
'assignvariableop_31_adam_enc_1_kernel_v:
4
%assignvariableop_32_adam_enc_1_bias_v:	;
'assignvariableop_33_adam_enc_2_kernel_v:
Ä4
%assignvariableop_34_adam_enc_2_bias_v:	Ä:
'assignvariableop_35_adam_enc_3_kernel_v:	Äb3
%assignvariableop_36_adam_enc_3_bias_v:b:
'assignvariableop_37_adam_dec_3_kernel_v:	bÄ4
%assignvariableop_38_adam_dec_3_bias_v:	Ä;
'assignvariableop_39_adam_dec_2_kernel_v:
Ä4
%assignvariableop_40_adam_dec_2_bias_v:	D
0assignvariableop_41_adam_decoder_output_kernel_v:
=
.assignvariableop_42_adam_decoder_output_bias_v:	
identity_44¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9º
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*à
valueÖBÓ,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ý
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Æ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_enc_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_enc_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_enc_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_enc_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_enc_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_enc_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_dec_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dec_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_dec_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dec_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp)assignvariableop_10_decoder_output_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp'assignvariableop_11_decoder_output_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_enc_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_enc_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_enc_2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_enc_2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_enc_3_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_enc_3_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dec_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dec_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dec_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dec_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_29AssignVariableOp0assignvariableop_29_adam_decoder_output_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_decoder_output_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_enc_1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_enc_1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_enc_2_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_enc_2_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_enc_3_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_enc_3_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dec_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_dec_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dec_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_dec_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_41AssignVariableOp0assignvariableop_41_adam_decoder_output_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp.assignvariableop_42_adam_decoder_output_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_44IdentityIdentity_43:output:0^NoOp_1*
T0*
_output_shapes
: î
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_44Identity_44:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
öW
¦
__inference__traced_save_110483
file_prefix+
'savev2_enc_1_kernel_read_readvariableop)
%savev2_enc_1_bias_read_readvariableop+
'savev2_enc_2_kernel_read_readvariableop)
%savev2_enc_2_bias_read_readvariableop+
'savev2_enc_3_kernel_read_readvariableop)
%savev2_enc_3_bias_read_readvariableop+
'savev2_dec_3_kernel_read_readvariableop)
%savev2_dec_3_bias_read_readvariableop+
'savev2_dec_2_kernel_read_readvariableop)
%savev2_dec_2_bias_read_readvariableop4
0savev2_decoder_output_kernel_read_readvariableop2
.savev2_decoder_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_enc_1_kernel_m_read_readvariableop0
,savev2_adam_enc_1_bias_m_read_readvariableop2
.savev2_adam_enc_2_kernel_m_read_readvariableop0
,savev2_adam_enc_2_bias_m_read_readvariableop2
.savev2_adam_enc_3_kernel_m_read_readvariableop0
,savev2_adam_enc_3_bias_m_read_readvariableop2
.savev2_adam_dec_3_kernel_m_read_readvariableop0
,savev2_adam_dec_3_bias_m_read_readvariableop2
.savev2_adam_dec_2_kernel_m_read_readvariableop0
,savev2_adam_dec_2_bias_m_read_readvariableop;
7savev2_adam_decoder_output_kernel_m_read_readvariableop9
5savev2_adam_decoder_output_bias_m_read_readvariableop2
.savev2_adam_enc_1_kernel_v_read_readvariableop0
,savev2_adam_enc_1_bias_v_read_readvariableop2
.savev2_adam_enc_2_kernel_v_read_readvariableop0
,savev2_adam_enc_2_bias_v_read_readvariableop2
.savev2_adam_enc_3_kernel_v_read_readvariableop0
,savev2_adam_enc_3_bias_v_read_readvariableop2
.savev2_adam_dec_3_kernel_v_read_readvariableop0
,savev2_adam_dec_3_bias_v_read_readvariableop2
.savev2_adam_dec_2_kernel_v_read_readvariableop0
,savev2_adam_dec_2_bias_v_read_readvariableop;
7savev2_adam_decoder_output_kernel_v_read_readvariableop9
5savev2_adam_decoder_output_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ·
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*à
valueÖBÓ,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÅ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B é
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_enc_1_kernel_read_readvariableop%savev2_enc_1_bias_read_readvariableop'savev2_enc_2_kernel_read_readvariableop%savev2_enc_2_bias_read_readvariableop'savev2_enc_3_kernel_read_readvariableop%savev2_enc_3_bias_read_readvariableop'savev2_dec_3_kernel_read_readvariableop%savev2_dec_3_bias_read_readvariableop'savev2_dec_2_kernel_read_readvariableop%savev2_dec_2_bias_read_readvariableop0savev2_decoder_output_kernel_read_readvariableop.savev2_decoder_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_enc_1_kernel_m_read_readvariableop,savev2_adam_enc_1_bias_m_read_readvariableop.savev2_adam_enc_2_kernel_m_read_readvariableop,savev2_adam_enc_2_bias_m_read_readvariableop.savev2_adam_enc_3_kernel_m_read_readvariableop,savev2_adam_enc_3_bias_m_read_readvariableop.savev2_adam_dec_3_kernel_m_read_readvariableop,savev2_adam_dec_3_bias_m_read_readvariableop.savev2_adam_dec_2_kernel_m_read_readvariableop,savev2_adam_dec_2_bias_m_read_readvariableop7savev2_adam_decoder_output_kernel_m_read_readvariableop5savev2_adam_decoder_output_bias_m_read_readvariableop.savev2_adam_enc_1_kernel_v_read_readvariableop,savev2_adam_enc_1_bias_v_read_readvariableop.savev2_adam_enc_2_kernel_v_read_readvariableop,savev2_adam_enc_2_bias_v_read_readvariableop.savev2_adam_enc_3_kernel_v_read_readvariableop,savev2_adam_enc_3_bias_v_read_readvariableop.savev2_adam_dec_3_kernel_v_read_readvariableop,savev2_adam_dec_3_bias_v_read_readvariableop.savev2_adam_dec_2_kernel_v_read_readvariableop,savev2_adam_dec_2_bias_v_read_readvariableop7savev2_adam_decoder_output_kernel_v_read_readvariableop5savev2_adam_decoder_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ô
_input_shapesâ
ß: :
::
Ä:Ä:	Äb:b:	bÄ:Ä:
Ä::
:: : : : : : : :
::
Ä:Ä:	Äb:b:	bÄ:Ä:
Ä::
::
::
Ä:Ä:	Äb:b:	bÄ:Ä:
Ä::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
Ä:!

_output_shapes	
:Ä:%!

_output_shapes
:	Äb: 

_output_shapes
:b:%!

_output_shapes
:	bÄ:!

_output_shapes	
:Ä:&	"
 
_output_shapes
:
Ä:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
Ä:!

_output_shapes	
:Ä:%!

_output_shapes
:	Äb: 

_output_shapes
:b:%!

_output_shapes
:	bÄ:!

_output_shapes	
:Ä:&"
 
_output_shapes
:
Ä:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::& "
 
_output_shapes
:
:!!

_output_shapes	
::&""
 
_output_shapes
:
Ä:!#

_output_shapes	
:Ä:%$!

_output_shapes
:	Äb: %

_output_shapes
:b:%&!

_output_shapes
:	bÄ:!'

_output_shapes	
:Ä:&("
 
_output_shapes
:
Ä:!)

_output_shapes	
::&*"
 
_output_shapes
:
:!+

_output_shapes	
::,

_output_shapes
: 
¤

õ
A__inference_enc_2_layer_call_and_return_conditional_losses_109673

inputs2
matmul_readvariableop_resource:
Ä.
biasadd_readvariableop_resource:	Ä
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¸
(__inference_model_2_layer_call_fn_109956
input_layer
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:	Äb
	unknown_4:b
	unknown_5:	bÄ
	unknown_6:	Ä
	unknown_7:
Ä
	unknown_8:	
	unknown_9:


unknown_10:	
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_109900p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
¤

õ
A__inference_enc_1_layer_call_and_return_conditional_losses_109656

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

ô
A__inference_dec_3_layer_call_and_return_conditional_losses_109707

inputs1
matmul_readvariableop_resource:	bÄ.
biasadd_readvariableop_resource:	Ä
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	bÄ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿb: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 
_user_specified_nameinputs
Þ3
	
C__inference_model_2_layer_call_and_return_conditional_losses_110180

inputs8
$enc_1_matmul_readvariableop_resource:
4
%enc_1_biasadd_readvariableop_resource:	8
$enc_2_matmul_readvariableop_resource:
Ä4
%enc_2_biasadd_readvariableop_resource:	Ä7
$enc_3_matmul_readvariableop_resource:	Äb3
%enc_3_biasadd_readvariableop_resource:b7
$dec_3_matmul_readvariableop_resource:	bÄ4
%dec_3_biasadd_readvariableop_resource:	Ä8
$dec_2_matmul_readvariableop_resource:
Ä4
%dec_2_biasadd_readvariableop_resource:	A
-decoder_output_matmul_readvariableop_resource:
=
.decoder_output_biasadd_readvariableop_resource:	
identity¢dec_2/BiasAdd/ReadVariableOp¢dec_2/MatMul/ReadVariableOp¢dec_3/BiasAdd/ReadVariableOp¢dec_3/MatMul/ReadVariableOp¢%decoder_output/BiasAdd/ReadVariableOp¢$decoder_output/MatMul/ReadVariableOp¢enc_1/BiasAdd/ReadVariableOp¢enc_1/MatMul/ReadVariableOp¢enc_2/BiasAdd/ReadVariableOp¢enc_2/MatMul/ReadVariableOp¢enc_3/BiasAdd/ReadVariableOp¢enc_3/MatMul/ReadVariableOp
enc_1/MatMul/ReadVariableOpReadVariableOp$enc_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0v
enc_1/MatMulMatMulinputs#enc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_1/BiasAdd/ReadVariableOpReadVariableOp%enc_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
enc_1/BiasAddBiasAddenc_1/MatMul:product:0$enc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

enc_1/ReluReluenc_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_2/MatMul/ReadVariableOpReadVariableOp$enc_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0
enc_2/MatMulMatMulenc_1/Relu:activations:0#enc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
enc_2/BiasAdd/ReadVariableOpReadVariableOp%enc_2_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0
enc_2/BiasAddBiasAddenc_2/MatMul:product:0$enc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ]

enc_2/ReluReluenc_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
enc_3/MatMul/ReadVariableOpReadVariableOp$enc_3_matmul_readvariableop_resource*
_output_shapes
:	Äb*
dtype0
enc_3/MatMulMatMulenc_2/Relu:activations:0#enc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb~
enc_3/BiasAdd/ReadVariableOpReadVariableOp%enc_3_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0
enc_3/BiasAddBiasAddenc_3/MatMul:product:0$enc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb\

enc_3/ReluReluenc_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dec_3/MatMul/ReadVariableOpReadVariableOp$dec_3_matmul_readvariableop_resource*
_output_shapes
:	bÄ*
dtype0
dec_3/MatMulMatMulenc_3/Relu:activations:0#dec_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
dec_3/BiasAdd/ReadVariableOpReadVariableOp%dec_3_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0
dec_3/BiasAddBiasAdddec_3/MatMul:product:0$dec_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ]

dec_3/ReluReludec_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
dec_2/MatMul/ReadVariableOpReadVariableOp$dec_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0
dec_2/MatMulMatMuldec_3/Relu:activations:0#dec_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dec_2/BiasAdd/ReadVariableOpReadVariableOp%dec_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dec_2/BiasAddBiasAdddec_2/MatMul:product:0$dec_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

dec_2/ReluReludec_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
decoder_output/MatMulMatMuldec_2/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¤
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydecoder_output/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
NoOpNoOp^dec_2/BiasAdd/ReadVariableOp^dec_2/MatMul/ReadVariableOp^dec_3/BiasAdd/ReadVariableOp^dec_3/MatMul/ReadVariableOp&^decoder_output/BiasAdd/ReadVariableOp%^decoder_output/MatMul/ReadVariableOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp^enc_3/BiasAdd/ReadVariableOp^enc_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2<
dec_2/BiasAdd/ReadVariableOpdec_2/BiasAdd/ReadVariableOp2:
dec_2/MatMul/ReadVariableOpdec_2/MatMul/ReadVariableOp2<
dec_3/BiasAdd/ReadVariableOpdec_3/BiasAdd/ReadVariableOp2:
dec_3/MatMul/ReadVariableOpdec_3/MatMul/ReadVariableOp2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2L
$decoder_output/MatMul/ReadVariableOp$decoder_output/MatMul/ReadVariableOp2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2<
enc_3/BiasAdd/ReadVariableOpenc_3/BiasAdd/ReadVariableOp2:
enc_3/MatMul/ReadVariableOpenc_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã

&__inference_enc_2_layer_call_fn_110240

inputs
unknown:
Ä
	unknown_0:	Ä
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_2_layer_call_and_return_conditional_losses_109673p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò;
´

!__inference__wrapped_model_109638
input_layer@
,model_2_enc_1_matmul_readvariableop_resource:
<
-model_2_enc_1_biasadd_readvariableop_resource:	@
,model_2_enc_2_matmul_readvariableop_resource:
Ä<
-model_2_enc_2_biasadd_readvariableop_resource:	Ä?
,model_2_enc_3_matmul_readvariableop_resource:	Äb;
-model_2_enc_3_biasadd_readvariableop_resource:b?
,model_2_dec_3_matmul_readvariableop_resource:	bÄ<
-model_2_dec_3_biasadd_readvariableop_resource:	Ä@
,model_2_dec_2_matmul_readvariableop_resource:
Ä<
-model_2_dec_2_biasadd_readvariableop_resource:	I
5model_2_decoder_output_matmul_readvariableop_resource:
E
6model_2_decoder_output_biasadd_readvariableop_resource:	
identity¢$model_2/dec_2/BiasAdd/ReadVariableOp¢#model_2/dec_2/MatMul/ReadVariableOp¢$model_2/dec_3/BiasAdd/ReadVariableOp¢#model_2/dec_3/MatMul/ReadVariableOp¢-model_2/decoder_output/BiasAdd/ReadVariableOp¢,model_2/decoder_output/MatMul/ReadVariableOp¢$model_2/enc_1/BiasAdd/ReadVariableOp¢#model_2/enc_1/MatMul/ReadVariableOp¢$model_2/enc_2/BiasAdd/ReadVariableOp¢#model_2/enc_2/MatMul/ReadVariableOp¢$model_2/enc_3/BiasAdd/ReadVariableOp¢#model_2/enc_3/MatMul/ReadVariableOp
#model_2/enc_1/MatMul/ReadVariableOpReadVariableOp,model_2_enc_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
model_2/enc_1/MatMulMatMulinput_layer+model_2/enc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model_2/enc_1/BiasAdd/ReadVariableOpReadVariableOp-model_2_enc_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¡
model_2/enc_1/BiasAddBiasAddmodel_2/enc_1/MatMul:product:0,model_2/enc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
model_2/enc_1/ReluRelumodel_2/enc_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model_2/enc_2/MatMul/ReadVariableOpReadVariableOp,model_2_enc_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0 
model_2/enc_2/MatMulMatMul model_2/enc_1/Relu:activations:0+model_2/enc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
$model_2/enc_2/BiasAdd/ReadVariableOpReadVariableOp-model_2_enc_2_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0¡
model_2/enc_2/BiasAddBiasAddmodel_2/enc_2/MatMul:product:0,model_2/enc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄm
model_2/enc_2/ReluRelumodel_2/enc_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
#model_2/enc_3/MatMul/ReadVariableOpReadVariableOp,model_2_enc_3_matmul_readvariableop_resource*
_output_shapes
:	Äb*
dtype0
model_2/enc_3/MatMulMatMul model_2/enc_2/Relu:activations:0+model_2/enc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
$model_2/enc_3/BiasAdd/ReadVariableOpReadVariableOp-model_2_enc_3_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0 
model_2/enc_3/BiasAddBiasAddmodel_2/enc_3/MatMul:product:0,model_2/enc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbl
model_2/enc_3/ReluRelumodel_2/enc_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
#model_2/dec_3/MatMul/ReadVariableOpReadVariableOp,model_2_dec_3_matmul_readvariableop_resource*
_output_shapes
:	bÄ*
dtype0 
model_2/dec_3/MatMulMatMul model_2/enc_3/Relu:activations:0+model_2/dec_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
$model_2/dec_3/BiasAdd/ReadVariableOpReadVariableOp-model_2_dec_3_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0¡
model_2/dec_3/BiasAddBiasAddmodel_2/dec_3/MatMul:product:0,model_2/dec_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄm
model_2/dec_3/ReluRelumodel_2/dec_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
#model_2/dec_2/MatMul/ReadVariableOpReadVariableOp,model_2_dec_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0 
model_2/dec_2/MatMulMatMul model_2/dec_3/Relu:activations:0+model_2/dec_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model_2/dec_2/BiasAdd/ReadVariableOpReadVariableOp-model_2_dec_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¡
model_2/dec_2/BiasAddBiasAddmodel_2/dec_2/MatMul:product:0,model_2/dec_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
model_2/dec_2/ReluRelumodel_2/dec_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
,model_2/decoder_output/MatMul/ReadVariableOpReadVariableOp5model_2_decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0²
model_2/decoder_output/MatMulMatMul model_2/dec_2/Relu:activations:04model_2/decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
-model_2/decoder_output/BiasAdd/ReadVariableOpReadVariableOp6model_2_decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¼
model_2/decoder_output/BiasAddBiasAdd'model_2/decoder_output/MatMul:product:05model_2/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_2/decoder_output/SigmoidSigmoid'model_2/decoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
IdentityIdentity"model_2/decoder_output/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
NoOpNoOp%^model_2/dec_2/BiasAdd/ReadVariableOp$^model_2/dec_2/MatMul/ReadVariableOp%^model_2/dec_3/BiasAdd/ReadVariableOp$^model_2/dec_3/MatMul/ReadVariableOp.^model_2/decoder_output/BiasAdd/ReadVariableOp-^model_2/decoder_output/MatMul/ReadVariableOp%^model_2/enc_1/BiasAdd/ReadVariableOp$^model_2/enc_1/MatMul/ReadVariableOp%^model_2/enc_2/BiasAdd/ReadVariableOp$^model_2/enc_2/MatMul/ReadVariableOp%^model_2/enc_3/BiasAdd/ReadVariableOp$^model_2/enc_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2L
$model_2/dec_2/BiasAdd/ReadVariableOp$model_2/dec_2/BiasAdd/ReadVariableOp2J
#model_2/dec_2/MatMul/ReadVariableOp#model_2/dec_2/MatMul/ReadVariableOp2L
$model_2/dec_3/BiasAdd/ReadVariableOp$model_2/dec_3/BiasAdd/ReadVariableOp2J
#model_2/dec_3/MatMul/ReadVariableOp#model_2/dec_3/MatMul/ReadVariableOp2^
-model_2/decoder_output/BiasAdd/ReadVariableOp-model_2/decoder_output/BiasAdd/ReadVariableOp2\
,model_2/decoder_output/MatMul/ReadVariableOp,model_2/decoder_output/MatMul/ReadVariableOp2L
$model_2/enc_1/BiasAdd/ReadVariableOp$model_2/enc_1/BiasAdd/ReadVariableOp2J
#model_2/enc_1/MatMul/ReadVariableOp#model_2/enc_1/MatMul/ReadVariableOp2L
$model_2/enc_2/BiasAdd/ReadVariableOp$model_2/enc_2/BiasAdd/ReadVariableOp2J
#model_2/enc_2/MatMul/ReadVariableOp#model_2/enc_2/MatMul/ReadVariableOp2L
$model_2/enc_3/BiasAdd/ReadVariableOp$model_2/enc_3/BiasAdd/ReadVariableOp2J
#model_2/enc_3/MatMul/ReadVariableOp#model_2/enc_3/MatMul/ReadVariableOp:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
¬

þ
J__inference_decoder_output_layer_call_and_return_conditional_losses_110331

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
´
C__inference_model_2_layer_call_and_return_conditional_losses_109900

inputs 
enc_1_109869:

enc_1_109871:	 
enc_2_109874:
Ä
enc_2_109876:	Ä
enc_3_109879:	Äb
enc_3_109881:b
dec_3_109884:	bÄ
dec_3_109886:	Ä 
dec_2_109889:
Ä
dec_2_109891:	)
decoder_output_109894:
$
decoder_output_109896:	
identity¢dec_2/StatefulPartitionedCall¢dec_3/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢enc_3/StatefulPartitionedCallå
enc_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_1_109869enc_1_109871*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_1_layer_call_and_return_conditional_losses_109656
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_109874enc_2_109876*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_2_layer_call_and_return_conditional_losses_109673
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_109879enc_3_109881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_3_layer_call_and_return_conditional_losses_109690
dec_3/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0dec_3_109884dec_3_109886*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_3_layer_call_and_return_conditional_losses_109707
dec_2/StatefulPartitionedCallStatefulPartitionedCall&dec_3/StatefulPartitionedCall:output:0dec_2_109889dec_2_109891*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_2_layer_call_and_return_conditional_losses_109724©
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall&dec_2/StatefulPartitionedCall:output:0decoder_output_109894decoder_output_109896*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_109741
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dec_2/StatefulPartitionedCall^dec_3/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dec_2/StatefulPartitionedCalldec_2/StatefulPartitionedCall2>
dec_3/StatefulPartitionedCalldec_3/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù

³
(__inference_model_2_layer_call_fn_110088

inputs
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:	Äb
	unknown_4:b
	unknown_5:	bÄ
	unknown_6:	Ä
	unknown_7:
Ä
	unknown_8:	
	unknown_9:


unknown_10:	
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_109900p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú
¹
C__inference_model_2_layer_call_and_return_conditional_losses_109990
input_layer 
enc_1_109959:

enc_1_109961:	 
enc_2_109964:
Ä
enc_2_109966:	Ä
enc_3_109969:	Äb
enc_3_109971:b
dec_3_109974:	bÄ
dec_3_109976:	Ä 
dec_2_109979:
Ä
dec_2_109981:	)
decoder_output_109984:
$
decoder_output_109986:	
identity¢dec_2/StatefulPartitionedCall¢dec_3/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢enc_3/StatefulPartitionedCallê
enc_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerenc_1_109959enc_1_109961*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_1_layer_call_and_return_conditional_losses_109656
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_109964enc_2_109966*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_2_layer_call_and_return_conditional_losses_109673
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_109969enc_3_109971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_3_layer_call_and_return_conditional_losses_109690
dec_3/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0dec_3_109974dec_3_109976*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_3_layer_call_and_return_conditional_losses_109707
dec_2/StatefulPartitionedCallStatefulPartitionedCall&dec_3/StatefulPartitionedCall:output:0dec_2_109979dec_2_109981*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_2_layer_call_and_return_conditional_losses_109724©
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall&dec_2/StatefulPartitionedCall:output:0decoder_output_109984decoder_output_109986*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_109741
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dec_2/StatefulPartitionedCall^dec_3/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dec_2/StatefulPartitionedCalldec_2/StatefulPartitionedCall2>
dec_3/StatefulPartitionedCalldec_3/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall:U Q
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinput_layer
¬

þ
J__inference_decoder_output_layer_call_and_return_conditional_losses_109741

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù

³
(__inference_model_2_layer_call_fn_110059

inputs
unknown:

	unknown_0:	
	unknown_1:
Ä
	unknown_2:	Ä
	unknown_3:	Äb
	unknown_4:b
	unknown_5:	bÄ
	unknown_6:	Ä
	unknown_7:
Ä
	unknown_8:	
	unknown_9:


unknown_10:	
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_109748p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
´
C__inference_model_2_layer_call_and_return_conditional_losses_109748

inputs 
enc_1_109657:

enc_1_109659:	 
enc_2_109674:
Ä
enc_2_109676:	Ä
enc_3_109691:	Äb
enc_3_109693:b
dec_3_109708:	bÄ
dec_3_109710:	Ä 
dec_2_109725:
Ä
dec_2_109727:	)
decoder_output_109742:
$
decoder_output_109744:	
identity¢dec_2/StatefulPartitionedCall¢dec_3/StatefulPartitionedCall¢&decoder_output/StatefulPartitionedCall¢enc_1/StatefulPartitionedCall¢enc_2/StatefulPartitionedCall¢enc_3/StatefulPartitionedCallå
enc_1/StatefulPartitionedCallStatefulPartitionedCallinputsenc_1_109657enc_1_109659*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_1_layer_call_and_return_conditional_losses_109656
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_109674enc_2_109676*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_2_layer_call_and_return_conditional_losses_109673
enc_3/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_3_109691enc_3_109693*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_enc_3_layer_call_and_return_conditional_losses_109690
dec_3/StatefulPartitionedCallStatefulPartitionedCall&enc_3/StatefulPartitionedCall:output:0dec_3_109708dec_3_109710*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_3_layer_call_and_return_conditional_losses_109707
dec_2/StatefulPartitionedCallStatefulPartitionedCall&dec_3/StatefulPartitionedCall:output:0dec_2_109725dec_2_109727*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dec_2_layer_call_and_return_conditional_losses_109724©
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall&dec_2/StatefulPartitionedCall:output:0decoder_output_109742decoder_output_109744*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_109741
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dec_2/StatefulPartitionedCall^dec_3/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall^enc_1/StatefulPartitionedCall^enc_2/StatefulPartitionedCall^enc_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2>
dec_2/StatefulPartitionedCalldec_2/StatefulPartitionedCall2>
dec_3/StatefulPartitionedCalldec_3/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2>
enc_3/StatefulPartitionedCallenc_3/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ó
A__inference_enc_3_layer_call_and_return_conditional_losses_110271

inputs1
matmul_readvariableop_resource:	Äb-
biasadd_readvariableop_resource:b
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Äb*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:b*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿba
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿbw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÄ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
Þ3
	
C__inference_model_2_layer_call_and_return_conditional_losses_110134

inputs8
$enc_1_matmul_readvariableop_resource:
4
%enc_1_biasadd_readvariableop_resource:	8
$enc_2_matmul_readvariableop_resource:
Ä4
%enc_2_biasadd_readvariableop_resource:	Ä7
$enc_3_matmul_readvariableop_resource:	Äb3
%enc_3_biasadd_readvariableop_resource:b7
$dec_3_matmul_readvariableop_resource:	bÄ4
%dec_3_biasadd_readvariableop_resource:	Ä8
$dec_2_matmul_readvariableop_resource:
Ä4
%dec_2_biasadd_readvariableop_resource:	A
-decoder_output_matmul_readvariableop_resource:
=
.decoder_output_biasadd_readvariableop_resource:	
identity¢dec_2/BiasAdd/ReadVariableOp¢dec_2/MatMul/ReadVariableOp¢dec_3/BiasAdd/ReadVariableOp¢dec_3/MatMul/ReadVariableOp¢%decoder_output/BiasAdd/ReadVariableOp¢$decoder_output/MatMul/ReadVariableOp¢enc_1/BiasAdd/ReadVariableOp¢enc_1/MatMul/ReadVariableOp¢enc_2/BiasAdd/ReadVariableOp¢enc_2/MatMul/ReadVariableOp¢enc_3/BiasAdd/ReadVariableOp¢enc_3/MatMul/ReadVariableOp
enc_1/MatMul/ReadVariableOpReadVariableOp$enc_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0v
enc_1/MatMulMatMulinputs#enc_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_1/BiasAdd/ReadVariableOpReadVariableOp%enc_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
enc_1/BiasAddBiasAddenc_1/MatMul:product:0$enc_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

enc_1/ReluReluenc_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
enc_2/MatMul/ReadVariableOpReadVariableOp$enc_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0
enc_2/MatMulMatMulenc_1/Relu:activations:0#enc_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
enc_2/BiasAdd/ReadVariableOpReadVariableOp%enc_2_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0
enc_2/BiasAddBiasAddenc_2/MatMul:product:0$enc_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ]

enc_2/ReluReluenc_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
enc_3/MatMul/ReadVariableOpReadVariableOp$enc_3_matmul_readvariableop_resource*
_output_shapes
:	Äb*
dtype0
enc_3/MatMulMatMulenc_2/Relu:activations:0#enc_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb~
enc_3/BiasAdd/ReadVariableOpReadVariableOp%enc_3_biasadd_readvariableop_resource*
_output_shapes
:b*
dtype0
enc_3/BiasAddBiasAddenc_3/MatMul:product:0$enc_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb\

enc_3/ReluReluenc_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dec_3/MatMul/ReadVariableOpReadVariableOp$dec_3_matmul_readvariableop_resource*
_output_shapes
:	bÄ*
dtype0
dec_3/MatMulMatMulenc_3/Relu:activations:0#dec_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
dec_3/BiasAdd/ReadVariableOpReadVariableOp%dec_3_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype0
dec_3/BiasAddBiasAdddec_3/MatMul:product:0$dec_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ]

dec_3/ReluReludec_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
dec_2/MatMul/ReadVariableOpReadVariableOp$dec_2_matmul_readvariableop_resource* 
_output_shapes
:
Ä*
dtype0
dec_2/MatMulMatMuldec_3/Relu:activations:0#dec_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dec_2/BiasAdd/ReadVariableOpReadVariableOp%dec_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dec_2/BiasAddBiasAdddec_2/MatMul:product:0$dec_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]

dec_2/ReluReludec_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$decoder_output/MatMul/ReadVariableOpReadVariableOp-decoder_output_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
decoder_output/MatMulMatMuldec_2/Relu:activations:0,decoder_output/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¤
decoder_output/BiasAddBiasAdddecoder_output/MatMul:product:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentitydecoder_output/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
NoOpNoOp^dec_2/BiasAdd/ReadVariableOp^dec_2/MatMul/ReadVariableOp^dec_3/BiasAdd/ReadVariableOp^dec_3/MatMul/ReadVariableOp&^decoder_output/BiasAdd/ReadVariableOp%^decoder_output/MatMul/ReadVariableOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp^enc_3/BiasAdd/ReadVariableOp^enc_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : 2<
dec_2/BiasAdd/ReadVariableOpdec_2/BiasAdd/ReadVariableOp2:
dec_2/MatMul/ReadVariableOpdec_2/MatMul/ReadVariableOp2<
dec_3/BiasAdd/ReadVariableOpdec_3/BiasAdd/ReadVariableOp2:
dec_3/MatMul/ReadVariableOpdec_3/MatMul/ReadVariableOp2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2L
$decoder_output/MatMul/ReadVariableOp$decoder_output/MatMul/ReadVariableOp2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2<
enc_3/BiasAdd/ReadVariableOpenc_3/BiasAdd/ReadVariableOp2:
enc_3/MatMul/ReadVariableOpenc_3/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
D
input_layer5
serving_default_input_layer:0ÿÿÿÿÿÿÿÿÿC
decoder_output1
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:y
Ú
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
layer_with_weights-5
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
»

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
»

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
»

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
»

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
²
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_ratemompmqmr!ms"mt)mu*mv1mw2mx9my:mzv{v|v}v~!v"v)v*v1v2v9v:v"
	optimizer
v
0
1
2
3
!4
"5
)6
*7
18
29
910
:11"
trackable_list_wrapper
v
0
1
2
3
!4
"5
)6
*7
18
29
910
:11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î2ë
(__inference_model_2_layer_call_fn_109775
(__inference_model_2_layer_call_fn_110059
(__inference_model_2_layer_call_fn_110088
(__inference_model_2_layer_call_fn_109956À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
C__inference_model_2_layer_call_and_return_conditional_losses_110134
C__inference_model_2_layer_call_and_return_conditional_losses_110180
C__inference_model_2_layer_call_and_return_conditional_losses_109990
C__inference_model_2_layer_call_and_return_conditional_losses_110024À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÐBÍ
!__inference__wrapped_model_109638input_layer"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Kserving_default"
signature_map
 :
2enc_1/kernel
:2
enc_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_enc_1_layer_call_fn_110220¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_enc_1_layer_call_and_return_conditional_losses_110231¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :
Ä2enc_2/kernel
:Ä2
enc_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_enc_2_layer_call_fn_110240¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_enc_2_layer_call_and_return_conditional_losses_110251¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	Äb2enc_3/kernel
:b2
enc_3/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_enc_3_layer_call_fn_110260¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_enc_3_layer_call_and_return_conditional_losses_110271¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	bÄ2dec_3/kernel
:Ä2
dec_3/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dec_3_layer_call_fn_110280¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dec_3_layer_call_and_return_conditional_losses_110291¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :
Ä2dec_2/kernel
:2
dec_2/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
­
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
Ð2Í
&__inference_dec_2_layer_call_fn_110300¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_dec_2_layer_call_and_return_conditional_losses_110311¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
):'
2decoder_output/kernel
": 2decoder_output/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_decoder_output_layer_call_fn_110320¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_decoder_output_layer_call_and_return_conditional_losses_110331¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
'
j0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÏBÌ
$__inference_signature_wrapper_110211input_layer"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	ktotal
	lcount
m	variables
n	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
k0
l1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
%:#
2Adam/enc_1/kernel/m
:2Adam/enc_1/bias/m
%:#
Ä2Adam/enc_2/kernel/m
:Ä2Adam/enc_2/bias/m
$:"	Äb2Adam/enc_3/kernel/m
:b2Adam/enc_3/bias/m
$:"	bÄ2Adam/dec_3/kernel/m
:Ä2Adam/dec_3/bias/m
%:#
Ä2Adam/dec_2/kernel/m
:2Adam/dec_2/bias/m
.:,
2Adam/decoder_output/kernel/m
':%2Adam/decoder_output/bias/m
%:#
2Adam/enc_1/kernel/v
:2Adam/enc_1/bias/v
%:#
Ä2Adam/enc_2/kernel/v
:Ä2Adam/enc_2/bias/v
$:"	Äb2Adam/enc_3/kernel/v
:b2Adam/enc_3/bias/v
$:"	bÄ2Adam/dec_3/kernel/v
:Ä2Adam/dec_3/bias/v
%:#
Ä2Adam/dec_2/kernel/v
:2Adam/dec_2/bias/v
.:,
2Adam/decoder_output/kernel/v
':%2Adam/decoder_output/bias/v­
!__inference__wrapped_model_109638!")*129:5¢2
+¢(
&#
input_layerÿÿÿÿÿÿÿÿÿ
ª "@ª=
;
decoder_output)&
decoder_outputÿÿÿÿÿÿÿÿÿ£
A__inference_dec_2_layer_call_and_return_conditional_losses_110311^120¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
&__inference_dec_2_layer_call_fn_110300Q120¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "ÿÿÿÿÿÿÿÿÿ¢
A__inference_dec_3_layer_call_and_return_conditional_losses_110291])*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿb
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÄ
 z
&__inference_dec_3_layer_call_fn_110280P)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿb
ª "ÿÿÿÿÿÿÿÿÿÄ¬
J__inference_decoder_output_layer_call_and_return_conditional_losses_110331^9:0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_decoder_output_layer_call_fn_110320Q9:0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
A__inference_enc_1_layer_call_and_return_conditional_losses_110231^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
&__inference_enc_1_layer_call_fn_110220Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
A__inference_enc_2_layer_call_and_return_conditional_losses_110251^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÄ
 {
&__inference_enc_2_layer_call_fn_110240Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÄ¢
A__inference_enc_3_layer_call_and_return_conditional_losses_110271]!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿb
 z
&__inference_enc_3_layer_call_fn_110260P!"0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "ÿÿÿÿÿÿÿÿÿb¼
C__inference_model_2_layer_call_and_return_conditional_losses_109990u!")*129:=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¼
C__inference_model_2_layer_call_and_return_conditional_losses_110024u!")*129:=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ·
C__inference_model_2_layer_call_and_return_conditional_losses_110134p!")*129:8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ·
C__inference_model_2_layer_call_and_return_conditional_losses_110180p!")*129:8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_model_2_layer_call_fn_109775h!")*129:=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_2_layer_call_fn_109956h!")*129:=¢:
3¢0
&#
input_layerÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_2_layer_call_fn_110059c!")*129:8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_model_2_layer_call_fn_110088c!")*129:8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¿
$__inference_signature_wrapper_110211!")*129:D¢A
¢ 
:ª7
5
input_layer&#
input_layerÿÿÿÿÿÿÿÿÿ"@ª=
;
decoder_output)&
decoder_outputÿÿÿÿÿÿÿÿÿ