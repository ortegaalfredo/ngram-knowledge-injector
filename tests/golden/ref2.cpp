// Golden reference: exact row-index math from qwen4exp.cpp set_input,
// using the REAL constants read out of Qwen3.8-Flash-Next-Q8_0.
#include <cstdint>
#include <cstdio>
#include <vector>
int main(){
    const int64_t ngram_size=3, per_gram=8, n_gram=3;
    const int64_t n_heads=(ngram_size-1)*per_gram;
    std::vector<uint64_t> m={23703573157769ULL,20109073645365ULL,8052911324071ULL};
    std::vector<uint64_t> hvs={20000003,20000023,20000033,20000047,20000059,20000063,
                               20000069,20000077,20000081,20000093,20000107,20000147,
                               20000153,20000159,20000161,20000171};
    std::vector<uint64_t> hof={0,20000003,40000026,60000059,80000106,100000165,120000228,
                               140000297,160000374,180000455,200000548,220000655,240000802,
                               260000955,280001114,300001275};
    const int64_t eos=248044;
    // token windows [tok, t-1, t-2]
    std::vector<std::vector<int64_t>> cases={
        {760,6511,314},{9338,369,11751},{13,760,6511},
        {eos,1,2},{1,2,eos},{0,0,0},{1,1,1},
        {248056,248056,248056},{2147483647,2147483646,1},
        {151644,9919,374},{248045,846,13},{100000,200000,300000},
        {777,248044,555},{777,248044,248044},{42,43,248044},
        {248044,248044,7},{999,248044,998},{5,248044,248044}
    };
    for(auto&c:cases){
        // ctx[0]=tok, ctx[1]=t-1, ctx[2]=t-2 with eos reset semantics
        std::vector<int64_t> ctx(3);
        ctx[0]=c[0];
        bool cut=false;
        for(int64_t s=1;s<3;s++){
            int64_t t=c[s];
            if(cut||t<0||t==eos){cut=true;ctx[s]=eos;} else ctx[s]=t;
        }
        printf("%lld %lld %lld:",(long long)c[0],(long long)c[1],(long long)c[2]);
        for(int64_t n=2;n<=n_gram;n++){
            uint64_t mixed=(uint64_t)ctx[0]*m[0];
            for(int64_t j=1;j<n;j++) mixed ^= (uint64_t)ctx[j]*m[j];
            int64_t base=(n-2)*per_gram;
            for(int64_t g=0;g<per_gram;g++){
                int64_t h=base+g;
                printf(" %d",(int32_t)(mixed % hvs[h] + hof[h]));
            }
        }
        printf("\n");
    }
    return 0;
}
